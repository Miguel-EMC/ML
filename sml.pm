package sml{
  use strict;
  use warnings;
  use Data::Dump qw(dump);
  use List::Util qw(zip min max sum uniq all any shuffle);
  use Tie::IxHash;
  use AI::MXNet qw(mx);
  use Chart::Plotly::Plot;
  use Chart::Plotly::Trace::Scatter;
  use Encode;
  use utf8; # Tell perl source code is utf-8
  binmode(STDOUT, ":utf8"); #Correcly prints Wide characters.
  
  # https://stackoverflow.com/questions/28373405/add-new-method-to-existing-object-in-perl
  sub add_to_class{ #@save
    # Register functions as methods in created class.
    my($class, $method_name, $code_ref) = @_; # $self, 
  
    {
      # We need to use symbolic references.
      no strict 'refs';
      no warnings;
      # Shove the code reference into the class' symbol table.
      *{$class.'::'.$method_name} = $code_ref;
    }
  }
  
  # La forma my ($arg1, $arg2) = @_; presupone siempre respetar una cantidad fija de parÃ¡metros. Si la cantidad es fija, use ese mÃ©todo.
  # Por otro lado, para una cantidad variable de parÃ¡metros use siempre la funciÃ³n d2l->get_arguments().
  # Ella se alimenta de los nombres de cada parÃ¡metro acompaÃ±ados de sus respectivos valores por defecto, ya que estÃ¡ conformada por una arreglo asociativo: key/value.
  # Al registrar los parÃ¡metros mediante la funciÃ³n d2l->get_arguments(), debe siempre respetar el orden correcto de los parÃ¡metros,
  # para que ella admita saltarse aquellos parÃ¡metros que se consideren innecesarios, ya que se alimentarÃ¡ con los valores definidos por defecto.
  # Al registrar los parÃ¡metros, considere: si en Python es None use el equivalente undef, si es True use 1, si es False use 0, etc.
  # Cuando no hay la definiciÃ³n del valor por defecto en el cÃ³digo Python, si el parÃ¡metro es de tipo string use la cadena vacÃ­a '',
  # si es numÃ©rico use 0, si es una referencia a un arreglo use [], si es un objeto como un tensor MXNet use undef.
  sub get_arguments{
    my ($self, $args, %named_args) = (shift, pop @_, ());
    tie my %args, 'Tie::IxHash';
    %args = @_; # Receives default keys/values
    
    # Tranfering key/value from @$args into %args
    for (my ($i, $key) = 0; $i < @$args; $i++){
      $key = $args->[$i];
      if (defined $key && ref(\$key) eq 'SCALAR' && exists $args{$key} && $key eq 'out'){ # Handling out as an exception, to preserve it in the input @_;
        $i++; # skips its value.
        next;
      }
      if (defined $key && ref(\$key) eq 'SCALAR' && exists $args{$key}){ # key validation.
        $args{$key} = $args->[$i + 1]; # updates default value
        $named_args{$key} = undef; # stores the key of an updated value
        splice @$args, $i, 2; # removes key and value from @$args
        $i--; # repositioning after removal
      }
    }
    
    # Updating default values of %args out of the first unnamed arguments present in @$args by their respective positions
    while (my ($key, $default_value, $new_value) = each %args){
      # print "$key\n";
      last if !@$args; # Exits if empty @$args
      next if exists $named_args{$key}; # Skips previously updated key/value
      # Left elements must be paired and even positions of @$args must be scalars to be considered as possible keys. In addition, $args->[2*$_] =~ /^[a-zA-Z]\w+$/ must be a variable type.
      last if @$args % 2 == 0 && all{ defined $args->[2*$_] && ref(\$args->[2*$_]) eq 'SCALAR' && $args->[2*$_] =~ /^[a-zA-Z]\w+$/} (0 .. (@$args / 2) -1);
      $new_value  = shift @$args;
      $args{$key} = $new_value if defined $new_value; # updates default value
    }
    
    # Handling additional arguments left in @$args at this point
    for (my ($i, $key) = 0; $i < @$args; $i++){
      $key = $args->[$i];
      if (defined $key && ref(\$key) eq 'SCALAR' && !exists $args{$key} && $key =~ /^[a-zA-Z]\w+$/){ # key validation
        # This block might still insert incorrect keys ?
        # print "New attribute found: $key\n";
        $args{$key} = $args->[$i + 1];
        $named_args{$key} = undef; # stores the key
        splice @$args, $i, 2; # removes key and value from @$args
        $i--; # repositioning after removal
      }
    }
    
    return %args;
  }
  
  sub pi{
    my $self = shift;
    return 4*atan2(1,1);
  }
  
  # Defined in Section 1.2.1 Load CSV File
  # Function for loading a CSV
  # Load a CSV file
  sub load_csv{
    my ($self, $file_path, %args) = (splice(@_, 0, 2), delimiter => '[,;\t]', @_);
    
    open (FILE, "<", $file_path) or die "Cannot open file $file_path: $!";
    my $header = <FILE>;
    chomp($header);
    my @dataset = ();
    while (<FILE>){
      my $row = $_;
      $row =~ s/[\r\n]+$//g; # Regular expression that deletes characters such as \r \n from a row
      next if (!defined $row || $row =~ /^\s*$/);
      push @dataset, [split /$args{delimiter}/, $row];
    }
    close FILE;
    
    return wantarray ? (\@dataset, $header) : \@dataset;
  }
  
  # Defined in Section 1.2.2 Convert String to Floats
  # Function For Converting String Data To Floats.
  # Convert string columns to float
  sub str_column_to_float{
    my ($self, $dataset, $column, %args) = (splice (@_, 0, 3), precision => 1, @_);
    
    return if ($dataset->[0][$column] !~ /^-?\d+/);
    
    $args{precision} = '%.' . $args{precision} . 'f';
    for my $row (@$dataset){
      $row->[$column] = sprintf ($args{precision}, $row->[$column]);
    }
  }
  
  
  # Defined in Section 1.2.3 Convert String to Integers
  # Function To Integer Encode String Class Values.
  # Convert string column to integer
  sub str_column_to_int{
    my ($self, $dataset, $column) = @_;
    my $class_values = [map {$_->[$column]} @$dataset];
    my @unique = uniq @$class_values;
    my %lookup = ();
    while (my ($i, $value) = each @unique) {
      $lookup{$value} = $i;
    }
    for my $row (@$dataset){
      $row->[$column] = $lookup{$row->[$column]};
    }
    return \%lookup;
  }
  
  sub type{
    my ($self, $var) = @_;
    
    return undef unless defined $var;
    if (ref $var) {
      return ref $var;
    }elsif ($var =~ /(-?\d+)(\.\d+)?/){
      return defined $2 ? 'Float' : 'Int';
    }elsif($var =~ /NaN/i){
      return 'NaN';
    }else{
      return 'Str';
    }
  }
  #my $var1 = 1;
  #my $var2 = '-1.0';
  #my $var3 = 'abc';
  #my $var4 = [1, 2, 3];
  #my $var5 = {'a' => 1};
  #
  #print type($var1); # Int
  #print type($var2); # Float
  #print type($var3); # Str
  #print type($var4); # ARRAY
  #print type($var5); # HASH
  #print dump type(my $var6); # undef
  
  # Defined in Section 2.2.1 Normalize Data
  # Function To Calculate the Min and Max Values For a Dataset.
  # Find the min and max values for each column
  sub dataset_minmax{
    my ($self, $dataset) = @_;
    
    if (ref($dataset) eq 'AI::MXNet::NDArray'){
      return mx->nd->stack($dataset->min(axis=>0), $dataset->max(axis=>0), axis=>1);
    }else{
      my @minmax;
      for my $i (0 .. $#{$dataset->[0]}){ # Be careful not to include the y labels
        my $col_values = [map {$_->[$i]}  @$dataset];
        my $value_min = min(@$col_values);
        my $value_max = max(@$col_values);
        push @minmax, [$value_min, $value_max];
      }
      return \@minmax;
    }
  }
  
  # Defined in Section 2.2.1 Normalize Data
  # Function To Normalize a Dataset.
  # Rescale dataset columns to the range 0-1
  sub normalize_dataset{
    my ($self, $dataset, $minmax) = @_;
    
    if (ref($dataset) eq 'AI::MXNet::NDArray'){
      # Extract min and max vectors
      my $min = $minmax->T->at(0);
      my $max = $minmax->T->at(1);
    
      # Broadcasting will automatically align shapes: (N, D) - (D,) = (N, D)
      (my $slice = $dataset->slice( [0, $dataset->shape->[0] -1], [0, $dataset->shape->[1] -1])) .= ($dataset - $min) / ($max - $min);
    }else{
      for my $row (@$dataset){
        for my $i (0 .. $#{$row}){ # Be careful not to include the y labels
          if (($minmax->[$i][1] - $minmax->[$i][0]) == 0) {
            $row->[$i] = ($row->[$i] - $minmax->[$i][0]) / 1;
          }else{
            $row->[$i] = ($row->[$i] - $minmax->[$i][0]) / ($minmax->[$i][1] - $minmax->[$i][0]);
          }
        }
      }
    }
  }
  
  # Defined in Section 2.2.2 Standardize Data
  # Function To Calculate Means For Each Column in a Dataset.
  # Calculate column means
  sub column_means{
    my ($self, $dataset) = @_;
    if (ref($dataset) eq 'AI::MXNet::NDArray'){
      return mx->nd->mean($dataset, axis=>0);
    }else{
      my $means = [0, map {$_} 0 .. $#{$dataset->[0]} -1];
      for my $i (0 .. $#{$dataset->[0]}){
        my $col_values = [map {$_->[$i]} @$dataset];
        $means->[$i] = sum(@$col_values) / scalar(@$dataset);
      }
      return $means;
    }
  }
  
  # Defined in Section 2.2.2 Standardize Data
  # Function To Calculate Standard Deviations For Each Column in a Dataset.
  # Calculate column standard deviations
  sub column_stdevs{
    my ($self, $dataset, $means) = @_;
    if (ref($dataset) eq 'AI::MXNet::NDArray'){
      return mx->nd->sqrt(($dataset - $means)->power(2)->sum(axis=>0) / ($dataset->len -1));
    }else{
      my $stdevs = [0, map {$_} 0 .. $#{$dataset->[0]} -1];
      for my $i (0 .. $#{$dataset->[0]}){
        my $variance = [map {($_->[$i] - $means->[$i]) ** 2} @$dataset];
        $stdevs->[$i] = sum(@$variance);
      }
      $stdevs = [map {sqrt($_ / (scalar(@$dataset) -1))} @$stdevs];
      return $stdevs;
    }
  }
  
  # Defined in Section 2.2.2 Standardize Data
  # Function To Standardize a Dataset.
  # Standardize dataset
  sub standardize_dataset{
    my ($self, $dataset, $means, $stdevs) = @_;
    if (ref($dataset) eq 'AI::MXNet::NDArray'){
      $dataset->slice([0, $dataset->shape->[0] -1], [0, $dataset->shape->[1] -1]) .= ($dataset - $means) / $stdevs;
    }else{
      for my $row (@$dataset){
        for my $i (0 .. $#$row){
          $row->[$i] = ($row->[$i] - $means->[$i]) / $stdevs->[$i];
        }
      }
    }
  }
  
  # Defined in Section 3.2.1 Train and Test Split
  # Function To Split a Dataset.
  # Split a dataset into a train and test set
  sub train_test_split{
    my ($self, $dataset, %args) = (splice (@_, 0, 2), split=>0.6, @_);
    
    if (ref($dataset) eq 'AI::MXNet::NDArray'){
      my $train_size = int($args{split} * $dataset->len);
      my $idx        = mx->nd->arange(stop => $dataset->len)->shuffle;
      my $train_idx  = $idx->slice(begin=>0, end=>$train_size);
      my $test_idx   = $idx->slice(begin=>$train_size, end=>$dataset->len);
      my $train      = mx->nd->take($dataset, $train_idx);
      my $test       = mx->nd->take($dataset, $test_idx);
      
      return $train, $test;
    }else{
      my $train_size = int($args{split} * @$dataset);
      my @idx        = shuffle (0 .. $#$dataset);
      my @train_idx  = @idx[0 .. $train_size -1];
      my @test_idx   = @idx[$train_size .. $#$dataset];
      my @train      = @$dataset[@train_idx];
      my @test       = @$dataset[@test_idx];
      
      return \@train, \@test;
    }
  }

  # Defined in Section 3.2.2 k-fold Cross-Validation Split
  # Function Create A Cross-Validation Split.
  # Split a dataset into $ k $ folds
  sub cross_validation_split{
    my ($self, $dataset, %args) = (splice (@_, 0, 2), n_folds=>10, @_);
    
    my @dataset_split;
    
    if (ref($dataset) eq 'AI::MXNet::NDArray'){
      my $fold_size  = int($dataset->len / $args{n_folds});
      my $idx        = mx->nd->arange(stop=>$dataset->len)->shuffle;
      for my $i (0 .. $args{n_folds} -1){
        my $fold_idx = $idx->slice(begin=>$i * $fold_size, end=>($i +1) * $fold_size);
        push @dataset_split, mx->nd->take($dataset, $fold_idx);
      }
      
      return mx->nd->stack(@dataset_split, axis=>0);
    }else{
      my $fold_size  = int(@$dataset / $args{n_folds});
      my @idx        = shuffle (0 .. $#$dataset);
      for my $i (0 .. $args{n_folds} -1){
        my @fold_idx = @idx[$i * $fold_size .. ($i +1) * $fold_size -1];
        push @dataset_split, [@$dataset[@fold_idx]];
      }
      return \@dataset_split;
    }
  }

=pod
# Example of a Cross-Validation Split of a Contrived Dataset.
# test cross validation split
mx->nd->random->seed(1);
my $dataset = mx->nd->array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]]);
my $folds = sml->cross_validation_split($dataset, n_folds=>4);
print $_->aspdl for @$folds;
# Example Output from Creating a Cross-Validation Split.
# [[[4], [6]], [[5], [3]], [[7], [10]], [[8], [1]]]
=cut

  sub count_labels{
    my ($self, $dataset) = @_;
    
    if (ref($dataset) eq 'AI::MXNet::NDArray'){
      my $Y = $dataset->slice_axis(axis=>1, begin=>-1, end=>$dataset->shape->[-1])->squeeze(axis=>1);
      my $num_classes = $Y->max->asscalar + 1;
      return mx->nd->one_hot($Y, $num_classes)->sum(axis=>0);
    }else{
      my %counts = ();
      map {$counts{"$_->[-1]"}++} @$dataset;
      return \%counts;
    }
  }

  # Defined in Section 4.2.1 Classification Accuracy
  # Function To Calculate Classification Accuracy.
  # Calculate accuracy percentage between two lists
  sub accuracy_metric{
    my ($self, $actual, $predicted) = @_;
    # Compute the number of correct predictions.
    
    if (ref($actual) eq 'AI::MXNet::NDArray'){
      
      if ($predicted->ndim > 1 && $predicted->shape->[1] > 1){ # from d2l.ai 3.6.5. Classification Accuracy
        $predicted = $predicted->argmax(axis => 1);
      }
      
      if($actual->size == $predicted->size && $actual->ndim != $predicted->ndim){
        $predicted = $predicted->reshape($actual->shape);
      }
        
      my $cmp = $predicted->astype($actual->dtype) == $actual;
      return sprintf '%0.2f', (100 * $cmp->astype($actual->dtype)->sum / $actual->len)->asscalar;
    }else{
      my $correct = 0;
      for my $pair (zip $actual, $predicted){
        $correct++ if ($pair->[0] == $pair->[1]);
      }
      return sprintf '%0.2f', $correct / @$actual * 100.0;
    }
  }
  
  # Defined in Section 4.2.2 Confusion Matrix
  # Function To Calculate a Confusion Matrix.
  # calculate a confusion matrix
  sub confusion_matrix{
    my ($self, $actual, $predicted) = @_;
    
    if (ref($actual) eq 'AI::MXNet::NDArray'){
      # Step 1: One-hot encode the actual and predicted arrays
      my $num_classes       = $actual->max->asscalar + 1;
      my $actual_one_hot    = mx->nd->one_hot($actual, $num_classes);
      my $predicted_one_hot = mx->nd->one_hot($predicted, $num_classes);
    
      # Step 2: Compute confusion matrix
      # Matrix multiplication: (actual_one_hot^T) * predicted_one_hot
      return mx->nd->arange(stop=>$num_classes), mx->nd->dot($actual_one_hot->T, $predicted_one_hot);
    }else{
      my @unique = uniq @$actual;
      my $matrix = [map {[]} 0 .. $#unique];
      for my $i (0 .. $#unique){
        $matrix->[$i] = [0, map {$_} 0 .. $#unique -1];
      }
      my (%lookup, $x, $y);
      while (my ($i, $value) = each @unique) {
        $lookup{$value} = $i;
      }
      for my $i (0 .. $#$actual){
        $x = $lookup{$actual->[$i]};
        $y = $lookup{$predicted->[$i]};
        $matrix->[$x][$y] += 1;
      }
      
      return \@unique, $matrix;
    }
  }
  
  # Defined in Section 4.2.2 Confusion Matrix
  # Function To Pretty Print a Confusion Matrix.
  # pretty print a confusion matrix
  sub print_confusion_matrix{
    my ($self, $unique, $matrix) = @_;
    if (ref($matrix) eq 'AI::MXNet::NDArray'){
      printf "A/P%s", $unique->aspdl;
      printf "%s", mx->nd->concat($unique->expand_dims(axis=>1), $matrix, dim=>1)->aspdl;
    }else{
      print 'A/P ' . join(' ', map {$_} @$unique), "\n";
      while (my ($i, $x) = each @$unique){
        print sprintf " %s| %s\n", $x, join(' ', map {$_} @{$matrix->[$i]});
      }
    }
  }

  # Defined in Section 4.2.3 Mean Absolute ErrorÂ¶
  # Function To Calculate Mean Absolute Error.
  # Calculate mean absolute error
  sub mae_metric{
    my ($self, $actual, $predicted) = @_;
    
    if (ref($actual) eq 'AI::MXNet::NDArray'){
      return sprintf '%0.2f', (mx->nd->abs($actual - $predicted)->sum / $actual->len)->asscalar;
    }else{
      my $sum_error = 0.0;
      for my $pair (zip $actual, $predicted){
        $sum_error += abs($pair->[0] - $pair->[1]);
      }
      return sprintf '%0.2f', $sum_error / @$actual;
    }
  }

  # Defined in Section 4.2.4 Root Mean Squared Error
  # Function To Calculate Root Mean Squared Error.
  # Calculate root mean squared error
  sub rmse_metric{
    my ($self, $actual, $predicted) = @_;
    
    if (ref($actual) eq 'AI::MXNet::NDArray'){
      if($actual->size == $predicted->size && $actual->ndim != $predicted->ndim){
        $predicted = $predicted->reshape($actual->shape);
      }
      my $prediction_error = ($predicted - $actual);
      my $sum_error = $prediction_error->square()->sum();
      my $mean_error = $sum_error / $actual->len;
      return sprintf '%0.3f', $mean_error->sqrt()->asscalar;
    }else{
      my $sum_error = 0.0;
      for my $pair (zip $actual, $predicted){
        $sum_error += (($pair->[0] - $pair->[1]) ** 2);
      }
      my $mean_error = $sum_error / @$actual;
      return sprintf '%0.3f', sqrt($mean_error);
    }
  }
  
  # Defined in Section 4.2.5 ROC curves
  # Function to calculate the ROC metrics
  sub perf_metrics{
    my ($self, $actual, $y_hat, $threshold) = @_;
    
    my ($tp, $fp, $tn, $fn, $tpr, $fpr) = (0, 0, 0, 0);
    if (ref($actual) eq 'AI::MXNet::NDArray'){
      # Step 1: Threshold to create binary predictions
      my $predicted = $y_hat >= $threshold;
    
      # Step 2: Convert actual and predicted to one-hot encoded matrices
      my $num_classes       = $actual->max->asscalar + 1;
      my $actual_one_hot    = mx->nd->one_hot($actual, $num_classes);    # Shape [n, $num_classes]
      my $predicted_one_hot = mx->nd->one_hot($predicted, $num_classes); # Shape [n, $num_classes]
    
      # Step 3: Compute confusion matrix using dot product
      my $confusion_matrix  = mx->nd->dot($actual_one_hot->T, $predicted_one_hot);
    
      # Extract counts from the confusion matrix
      $tp = $confusion_matrix->at(1)->at(1); # True Positives
      $fn = $confusion_matrix->at(1)->at(0); # False Negatives
      $fp = $confusion_matrix->at(0)->at(1); # False Positives
      $tn = $confusion_matrix->at(0)->at(0); # True Negatives
      
      # Step 4: Compute TPR and FPR
      $tpr = $tp / ($tp + $fn); # True Positive Rate
      $fpr = $fp / ($fp + $tn); # False Positive Rate
      
      return sprintf('%0.2f', $fpr->asscalar), sprintf('%0.2f', $tpr->asscalar);
    }else{
      # Step 1: Threshold to create binary predictions
      # my $predicted = [map { $_ >= $threshold ? 1 : 0 } @$y_hat];
      for my $i (0 .. $#$y_hat) {
        if ($y_hat->[$i] >= $threshold){
          if ($actual->[$i] == 1) {
            $tp++;
          }else {
            $fp++;
          }
        }else{
          if ($actual->[$i] == 0){
            $tn++;
          }else{
            $fn++;
          }
        }
      }
      
      # Step 4: Compute TPR and FPR
      $tpr = $tp / ($tp + $fn); # True Positive Rate
      $fpr = $fp / ($fp + $tn); # False Positive Rate
      
      return sprintf('%0.2f', $fpr), sprintf('%0.2f', $tpr);
    }
  }
  
  # Defined in Section 4.2.5 ROC curves
  # Function to calculate the integral using the trapezoid rule
  sub trapz{
    my ($self, $x, $y) = @_;
    
    if (ref($x) eq 'AI::MXNet::NDArray'){
      # Compute differences (x[i+1] - x[i])
      my $dx = $x->slice(begin=>1, end=>$x->shape->[-1]) 
             - $x->slice(begin=>0, end=>-1);
    
      # Compute averages of y values (y[i] + y[i+1]) / 2
      my $avg_y = ($y->slice(begin=>1, end=>$y->shape->[-1]) 
                 + $y->slice(begin=>0, end=>-1)) / 2;
    
      # Compute trapezoid areas and sum them
      return sprintf '%0.2f', mx->nd->sum($dx * $avg_y)->asscalar;
    }else{
      my $sum = 0;
      for my $i (0 .. @$x - 2){
        $sum += ($x->[$i + 1] - $x->[$i]) * ($y->[$i] + $y->[$i + 1]) / 2;
      }
      return sprintf '%0.2f', $sum;
    }
  }

  # Defined in Section 5.2.1 Random Prediction Algorithm
  # Example of Making Random Predictions
  # Generate random predictions
  sub random_algorithm{
    my ($self, $train, $test) = @_;
    if (ref($train) eq 'AI::MXNet::NDArray'){
      my $output_values = $train->slice_axis(axis=>1, begin=>-1, end=>$train->shape->[-1])->squeeze();
      return $output_values->shuffle->slice(begin=>0, end=>$test->len);
    }else{
      my @output_values = map {$_->[-1]} @$train;
      my @unique        = uniq @output_values;
      return [map { $unique[int(rand(@unique))] } @$test]; # predictions
    }
  }
    
  # Defined in Section 5.2.2 Zero Rule Algorithm: Classification
  # Function To Make Zero Rule Classification Predictions.
  # zero rule algorithm for classification
  sub zero_rule_algorithm_classification{
    my ($self, $train, $test) = @_;
    if (ref($train) eq 'AI::MXNet::NDArray'){
      my $output_values = $train->slice_axis(axis=>1, begin=>-1, end=>$train->shape->[-1])->squeeze();
      my $num_classes   = $output_values->max->asscalar + 1;
      my $counter       = mx->nd->one_hot($output_values, $num_classes)->sum(axis=>0);
      my $prediction    = mx->nd->argmax($counter); 
      return mx->nd->full([$test->len], $prediction->asscalar);
    }else{
      my @output_values = map {$_->[-1]} @$train;
      my %counter       = ();
      map {$counter{"$_"}++} @output_values; # Counts
      my $prediction    = $counter{max map {$_} %counter = reverse %counter}; 
      return [($prediction) x scalar(@$test)]; # predictions
    }
  }
 
  # Defined in Section 5.2.2 Zero Rule Algorithm: Regression
  # Function To Make Zero Rule Regression Predictions.
  # zero rule algorithm for regression
  sub zero_rule_algorithm_regression{
    my ($self, $train, $test) = @_;
    if (ref($train) eq 'AI::MXNet::NDArray'){
      my $output_values = $train->slice_axis(axis=>1, begin=>-1, end=>$train->shape->[-1])->squeeze();
      my $prediction    = $output_values->sum() / $output_values->len;
      return mx->nd->full([$test->len], $prediction->asscalar);
    }else{
      my @output_values = map {$_->[-1]} @$train;
      my $prediction    = sprintf '%0.1f', sum(@output_values) / @output_values;
      return [map {$prediction} (0 .. $#$test)]; # predictions
    }
  }
  
  # Defined in Section 6.2.1 Train-Test Algorithm Test Harness
  # Function To Evaluate An Algorithm Using a Train/Test Split.
  # Evaluate an algorithm using a train/test split
  sub evaluate_algorithm_train_test_split{
    my ($self, $dataset, $algorithm, %args) = ((splice @_, 0, 3), split=>0.6, metric=>undef, @_);
    
    my ($train, $test) = sml->train_test_split($dataset, split=>$args{split});
    my ($actual, $predicted, $score);
    
    if (ref($train) eq 'AI::MXNet::NDArray'){
      my $test_set = $test->copy();
      # (my $slice = $test_set->slice('X', -1)) .= mx->nd->full([$test_set->len, 1], 'NaN'); # The test labels are needed for plotting the loss curve
     
      $predicted = $algorithm->('sml', $train, $test_set, @_);
      $actual    = $test->slice_axis(axis=>1, begin=>-1, end=>$test->shape->[1])->squeeze();
    }else{
      my @test_set = ();
      for my $row (@$test){
        my @row_copy  = @$row;
        # $row_copy[-1] = undef; # The test labels are needed for plotting the loss curve
        push @test_set, [@row_copy];
      }
      $predicted = $algorithm->('sml', $train, \@test_set, @_);
      $actual    = [map {$_->[-1]} @$test];
    }
    
    # Regression : Classification
    if (defined $args{metric}){
      if ($args{metric} =~ /accuracy/i) {
        $score = sml->accuracy_metric($actual, $predicted);
      }elsif($args{metric} =~ /rmse/i){
        $score = sml->rmse_metric($actual, $predicted);
      }
    }elsif (ref($dataset) eq 'AI::MXNet::NDArray'){
      if (mx->nd->sum($actual->trunc() - $actual)->asscalar != 0){
        $score = sml->rmse_metric($actual, $predicted);
      }else{
        $score = sml->accuracy_metric($actual, $predicted);
      }
    }else{
      $score = (grep { $_ =~ /\d+\.\d+/} @$actual) ?
                sml->rmse_metric($actual, $predicted) :
                sml->accuracy_metric($actual, $predicted);
    }
    
    return wantarray ? ($score, $train, $test, $actual, $predicted) : $score;
  }

  # Defined in Section 6.2.2 Cross-Validation Algorithm Test Harness
  # Function To Evaluate An Algorithm Using k-fold Cross-Validation.
  # Evaluate an algorithm using a cross-validation split
  sub evaluate_algorithm_cross_validation_split{
    my ($self, $dataset, $algorithm, %args) = ((splice @_, 0, 3), n_folds=>10, metric=>undef, @_);
    
    my @folds = @{sml->cross_validation_split($dataset, n_folds=>$args{n_folds})};
    my ($actual, $predicted, $train_loss, $test_loss, @scores, @train_losses, @test_losses, @predictions, @actuals);
  
    while (my ($i, $fold) = each @folds){
      my @train_set = @folds;
      
      if (ref($dataset) eq 'AI::MXNet::NDArray'){
        my $test_set = (splice @train_set, $i, 1)->copy;
        # (my $slice = $test_set->slice('X', -1)) .= mx->nd->full([$test_set->len, 1], 'NaN'); # The test labels are needed for plotting the loss curve
        
        my $train_set = mx->nd->concat(@train_set, dim=>0); # print $test_set->shape;
        ($predicted, $train_loss, $test_loss) = $algorithm->('sml', $train_set, $test_set, %args);
        $actual = $fold->slice_axis(axis=>1, begin=>-1, end=>$fold->shape->[1])->squeeze();
      }else{
        splice @train_set, $i, 1;
        @train_set = map {@$_} @train_set;
        my @test_set = ();
        for my $row (@$fold){
          my @row_copy  = @$row;
          push @test_set, [@row_copy];
          # $row_copy[-1] = undef;# The test labels are needed for plotting the loss curve
        } # print scalar @test_set; 
        ($predicted, $train_loss, $test_loss) = $algorithm->('sml', \@train_set, \@test_set, %args);
        $actual = [map {$_->[-1]} @$fold];
      }
      
      # Regression : Classification
      if (defined $args{metric}){
        if ($args{metric} =~ /accuracy/i) {
          push @scores, sml->accuracy_metric($actual, $predicted);
        }elsif($args{metric} =~ /rmse/i){
          push @scores, sml->rmse_metric($actual, $predicted);
        }
      }elsif (ref($dataset) eq 'AI::MXNet::NDArray'){
        if (mx->nd->sum($actual->trunc() - $actual)->asscalar != 0){
          push @scores, sml->rmse_metric($actual, $predicted);
        }else{
          push @scores, sml->accuracy_metric($actual, $predicted);
        }
      }else{
        push @scores, (grep { $_ =~ /\d+\.\d+/} @$actual) ?
                       sml->rmse_metric($actual, $predicted) :
                       sml->accuracy_metric($actual, $predicted);
      }
      
      push @train_losses, $train_loss;
      push @test_losses, $test_loss;
      push @actuals, $actual;
      push @predictions, $predicted;
    }
    
    return wantarray ? (\@scores, \@train_losses, \@test_losses, \@actuals, \@predictions) : \@scores;
  }
  
  # Defined in Section 7.2.4 Make Predictions
  # Evaluate regression algorithm on training dataset
  sub evaluate_algorithm_no_split{
    my ($self, $dataset, $algorithm, %args) = (splice(@_, 0, 3), metric=>undef, @_);
    
    my ($actual, $predicted, $score);
    if (ref($dataset) eq 'AI::MXNet::NDArray'){
      my $test_set = $dataset->copy();
      # (my $slice = $test_set->slice('X', -1)) .= mx->nd->full([$test_set->len, 1], 'NaN'); # The test labels are needed for plotting the loss curve
      
      $predicted = $algorithm->('sml', $dataset, $test_set, @_);
      $actual    = $dataset->slice_axis(axis=>1, begin=>-1, end=>$dataset->shape->[1])->squeeze();
    }else{
      my @test_set = ();
      for my $row (@$dataset){
        my @row_copy  = @$row;
        # $row_copy[-1] = undef; # The test labels are needed for plotting the loss curve
        push @test_set, \@row_copy;
      }
      $predicted = $algorithm->('sml', $dataset, \@test_set, @_);
      $actual    = [map {$_->[-1]} @$dataset];
    }
    
    # Regression : Classification
    if (defined $args{metric}){
      if ($args{metric} =~ /accuracy/i) {
        $score = sml->accuracy_metric($actual, $predicted);
      }elsif($args{metric} =~ /rmse/i){
        $score = sml->rmse_metric($actual, $predicted);
      }
    }elsif (ref($dataset) eq 'AI::MXNet::NDArray'){
      if (mx->nd->sum($actual->trunc() - $actual)->asscalar != 0){
        $score = sml->rmse_metric($actual, $predicted);
      }else{
        $score = sml->accuracy_metric($actual, $predicted);
      }
    }else{
      $score = (grep { $_ =~ /\d+\.\d+/} @$actual) ?
                sml->rmse_metric($actual, $predicted) :
                sml->accuracy_metric($actual, $predicted);
    }
    
    return wantarray ? ($score, undef, $dataset, $actual, $predicted) : $score;
  }
  
  # Defined in Section 7.2.1 Calculate Mean and Variance
  # Calculate the mean value of a list of numbers
  # Function To Calculate the Mean of a List of Numbers.
  sub mean{
    my ($self, $values, %args) = (splice (@_, 0, 2), axis=>'None', @_);
    if (ref($values) eq 'AI::MXNet::NDArray'){
      return $values->sum(axis=>$args{axis}) / $values->len;
    }else{
      return sum(@$values) / scalar(@$values);
    }
  }
  
  # Defined in Section 7.2.1 Calculate Mean and Variance
  # Function to Calculate the Variance of a List of Numbers.
  # Calculate the variance of a list of numbers
  sub variance{
    my ($self, $values, $mean) = @_;
    if (ref($values) eq 'AI::MXNet::NDArray'){
      return ($values - $mean)->power(2)->sum();
    }else{
      return sum(map{($_ - $mean) ** 2} @$values);
    }
  }
  
  # Defined in Section 7.2.2 Calculate Covariance
  # Function to Calculate the Covariance.
  # Calculate covariance between x and y
  sub covariance{
    my ($self, $x, $mean_x, $y, $mean_y) = @_;
    
    if (ref($x) eq 'AI::MXNet::NDArray'){
      return (($x - $mean_x) * ($y - $mean_y))->sum();
    }else{
      my $covar = 0.0;
      for (my $i = 0; $i < @$x; $i++){
        $covar += ($x->[$i] - $mean_x) * ($y->[$i] - $mean_y);
      }
      return $covar;
    }
  }
  
  # Defined in Section 7.2.3 Estimate Coefficients
  # Calculate coefficients
  # Function To Calculate the Coefficients.
  sub coefficients{
    my ($self, $dataset) = @_;
    
    if (ref($dataset) eq 'AI::MXNet::NDArray'){
      my ($X, $Y) = @{$dataset->T};
      my ($x_mean, $y_mean) = (sml->mean($X), sml->mean($Y));
      my $b1 = sml->covariance($X, $x_mean, $Y, $y_mean) / sml->variance($X, $x_mean);
      my $b0 = $y_mean - $b1 * $x_mean;
      return $b0, $b1;
    }else{
      my ($X, $Y) = (zip @$dataset);
      my ($x_mean, $y_mean) = (sml->mean($X), sml->mean($Y));
      my $b1 = sml->covariance($X, $x_mean, $Y, $y_mean) / sml->variance($X, $x_mean);
      my $b0 = $y_mean - $b1 * $x_mean;
      return $b0, $b1;
    }
  }
  
  # Defined in Section 7.2.4 Make Predictions
  # Function To Run Simple Linear Regression.
  sub simple_linear_regression{
    my ($self, $train, $test) = @_;
    
    if (ref($train) eq 'AI::MXNet::NDArray'){
      my ($b0, $b1) = sml->coefficients($train);
      return $b0 + $test->transpose->at(0) * $b1; # Reduction from 2 to 1 dimension.
    }else{
      my @predictions = ();
      my ($b0, $b1) = sml->coefficients($train);
      for my $row (@$test){
        my $yhat = $b0 + $b1 * $row->[0];
        push @predictions, $yhat;
      }
      return \@predictions;
    }
  }
  
  # Local definition only.
  # Choses evaluation method between either train/test or cross-validation split.
  # Depends on the input parameter name or value.
  sub evaluate_algorithm{
    my ($self, $dataset, $algorithm, %args) = (splice (@_, 0, 3), split=>undef, n_folds=>undef, metric=>undef, @_);

    if (!defined $args{split} && !defined $args{n_folds}){
      return sml->evaluate_algorithm_no_split($dataset, $algorithm, metric=>$args{metric}, @_);
    }else{
      if(defined $args{split}){
        return sml->evaluate_algorithm_train_test_split($dataset, $algorithm, split=>$args{split}, metric=>$args{metric}, @_);
      }elsif (defined $args{n_folds}){
        return sml->evaluate_algorithm_cross_validation_split($dataset, $algorithm, n_folds=>$args{n_folds}, metric=>$args{metric}, @_);
      }
    }
  }

  # Defined in Section 8.2.2 Estimating Coefficients
  # Function To Estimate Coefficients With Stochastic Gradient Descent.
  # Estimate linear regression coefficients using stochastic gradient descent
  sub coefficients_sgd_linear{
    my ($self, $train, $l_rate, $n_epoch, %args) = ((splice @_, 0, 4), test=>undef, @_);
    
    if (ref($train) eq 'AI::MXNet::NDArray'){
      my $num_features = $train->shape->[1] - 1; # Excluding the last column (target)
      my $coef = mx->nd->zeros([$num_features + 1]); # Coefficients, including intercept term
      my (@train_loss, @test_loss, $sum_error, $yhat, $error, $X, $y, $X_bias, $gradient);
      for my $epoch (1 .. $n_epoch) {
        # Separate features (X) and target (y) from training data
        $X = $train->slice_axis(axis=>1, begin=>0,  end=>-1);
        $y = $train->slice_axis(axis=>1, begin=>-1, end=>$train->shape->[1])->squeeze();
    
        # Add a bias term to X
        $X_bias = mx->nd->concat(mx->nd->ones([$X->len, 1]), $X, dim => 1);
    
        # Calculate predictions
        $yhat = mx->nd->dot($X_bias, $coef);
    
        # Calculate error
        $error = $yhat - $y;
        # Compute loss (sum of squared errors)
        $sum_error = mx->nd->sum($error ** 2);
    
        # Update coefficients using SGD
        $gradient = mx->nd->dot($X_bias->T, $error) / $X->len;
        $coef = $coef - $l_rate * $gradient;
    
        printf " >epoch=%d, lrate=%.3f, error=%.3f\n", $epoch, $l_rate, $sum_error->asscalar if $epoch % 10 == 0;
    
        # Optionally store loss for analysis
        push @train_loss, $sum_error;
    
        # Evaluate test loss if test data is provided
        if ($args{test}){
          my $X_test = $args{test}->slice_axis(axis=>1, begin=>0,  end=>-1);
          my $y_test = $args{test}->slice_axis(axis=>1, begin=>-1, end=>$args{test}->shape->[1])->squeeze();
          my $X_test_bias = mx->nd->concat(mx->nd->ones([$X_test->len, 1]), $X_test, dim => 1);
          my $yhat_test   = mx->nd->dot($X_test_bias, $coef);
          my $test_error  = mx->nd->sum(($yhat_test - $y_test) ** 2);
          push @test_loss, $test_error;
        }
      }
    
      return $coef, mx->nd->stack(@train_loss)->squeeze(), 
             ($args{test} ? mx->nd->stack(@test_loss)->squeeze() : undef);
    }else{
      my $num_features = scalar(@{$train->[0]}) - 1; # Excluding the last column (target)
      my $coef = [(0.0) x ($num_features + 1)]; # Coefficients, including intercept term
      my (@train_loss, @test_loss, $sum_error, $yhat, $expected, $error);
      for my $epoch (1 .. $n_epoch){
        $sum_error = 0.0;
        for my $row (@$train){
          # Compute prediction: yhat = bias + sum(w_i * x_i)
          $yhat = $coef->[0]; # bias term
          for my $i (0 .. $num_features -1){
            $yhat += $coef->[$i + 1] * $row->[$i];
          }
          $expected   = $row->[$num_features]; # last element is the target
          $error      = $yhat - $expected;
          $sum_error += $error ** 2;
          
          # Update bias
          $coef->[0] -= $l_rate * $error;
          
          # Update each feature weight
          for my $i (0 .. $num_features -1){
            $coef->[$i + 1] -= $l_rate * $error * $row->[$i];
          }
        }
        printf " >epoch=%d, lrate=%.5f, error=%.2f\n", $epoch, $l_rate, $sum_error if $epoch % 10 == 0;
        
        # Storing performance data for the plot:
        push @train_loss, $sum_error;
        
        # Evaluate test loss if test data is provided
        if ($args{test} && ref($args{test}) eq 'ARRAY'){
          $sum_error = 0.0;
          for my $row (@{$args{test}}){
            # Compute prediction: yhat = bias + sum(w_i * x_i)
            $yhat = $coef->[0]; # bias term
            for my $i (0 .. $num_features -1){
              $yhat += $coef->[$i + 1] * $row->[$i];
            }
            $error      = $row->[-1] - $yhat;
            $sum_error += $error ** 2;
          }
          push @test_loss, $sum_error;
        }
      }
      
      return $coef, \@train_loss, ($args{test} ? \@test_loss : undef);
    }
  }

  # Defined in Section 8.2.3 Wine Quality Case Study
  # Linear Regression Algorithm With Stochastic Gradient Descent
  sub linear_regression_sgd{
    my ($self, $train, $test, %args) = ((splice @_, 0, 3), l_rate  => undef, 
                                                           n_epoch => undef, @_);
    
    my ($coef, $train_loss, $test_loss) = sml->coefficients_sgd_linear($train, 
                                                                       $args{l_rate}, 
                                                                       $args{n_epoch}, 
                                                                       test   => $test,
                                                                       metric => $args{metric});
    if (ref($test) eq 'AI::MXNet::NDArray'){
      my $X      = $test->slice_axis(axis=>1, begin=>0, end=>-1); # Obtain X from dataset
      my $X_bias = mx->nd->concat(mx->nd->ones([$X->len, 1]), $X, dim => 1); # Add a bias term to X
      my $pred   = mx->nd->dot($X_bias, $coef); # Predictions (Y_hat = X * coeficientes)
      return $pred, $train_loss, $test_loss;
    }else{
      my @predictions = ();
      for my $row (@$test){
        my $num_features = scalar(@$row) - 1; # Excluding the last column (target)
        # Compute prediction: yhat = bias + sum(w_i * x_i)
        my $yhat = $coef->[0]; # bias term
        for my $i (0 .. $num_features -1){
          $yhat += $coef->[$i + 1] * $row->[$i];
        }
        push @predictions, $yhat;
      }
      return \@predictions, $train_loss, $test_loss;
    }
  }

  # Defined in Section 9.2.2 Estimating Coefficients
  # Function To Estimate Logistic Regression Coefficients.
  # Estimate logistic regression coefficients using stochastic gradient descent
  sub coefficients_sgd_logistic{
    my ($self, $train, $l_rate, $n_epoch, %args) = ((splice @_, 0, 4), test=>undef, @_);
    
    my $coef = [(0.0) x scalar(@{$train->[0]})];
    my (@train_loss, @test_loss, $sum_error, $yhat, $error);
    for my $epoch (0 .. $n_epoch-1){
      $sum_error = 0.0;
      for my $row (@$train){
        $yhat      = sml->predict($row, $coef);
        $error     = $row->[-1] - $yhat;
        $sum_error += $error ** 2;
        $coef->[0] = $coef->[0] + $l_rate * $error * $yhat * (1.0 - $yhat);
        for my $i (0 .. $#{$row}-1){
          $coef->[$i + 1] = $coef->[$i + 1] + $l_rate * $error * $yhat * (1.0 - $yhat) * $row->[$i];
        }
      }
      printf " >epoch=%d, lrate=%.3f, error=%.3f\n", $epoch, $l_rate, $sum_error;
      
      # Storing performance data for the plot:
      push @train_loss, $sum_error;
      if (defined $args{test}){
        unless (ref($args{test}) eq 'ARRAY'){
          print STDERR "Test data is defined but it is not an array reference.\n";
        }else{
          $sum_error = 0.0;
          for my $row (@{$args{test}}){
            $yhat       = sml->predict($row, $coef);
            $error      = $row->[-1] - $yhat;
            $sum_error  += $error ** 2;
          }
          push @test_loss, $sum_error;
        }
      }
    }
    
    return $coef, \@train_loss, \@test_loss;
  }
  
  # Defined in Section 9.2.3 Pima Indians Diabetes Case Study
  # Example of Logistic Regression Applied to the Diabetes Dataset.
  # Logistic Regression Algorithm With Stochastic Gradient Descent.
  sub logistic_regression{
    my ($self, $train, $test, %args) = ((splice @_, 0, 3), l_rate  => undef, 
                                                           n_epoch => undef, @_);
    
    my @predictions = ();
    my ($coef, $train_loss, $test_loss) = sml->coefficients_sgd_logistic($train, 
                                                                         $args{l_rate}, 
                                                                         $args{n_epoch}, 
                                                                         test=>$test);
    for my $row (@$test){
      my $yhat = sml->predict($row, $coef);
      push @predictions, sprintf '%.0f', $yhat;
    }
    
    return \@predictions, $train_loss, $test_loss;
  }

  # Defined in Section 10.2.2 Training Network Weights
  # Function To Estimate Weights for the Perceptron.
  # Estimate Perceptron weights using stochastic gradient descent
  sub train_weights{
    my ($self, $train, $l_rate, $n_epoch, %args) = ((splice @_, 0, 4), test=>undef, @_);
    
    my $weights  = [(0.0) x scalar(@{$train->[0]})];
    my (@train_loss, @test_loss, $sum_error, $yhat, $error);
    for my $epoch (0 .. $n_epoch-1){
      $sum_error = 0.0;
      for my $row (@$train){
        $yhat          = sml->predict($row, $weights);
        $error         = $row->[-1] - $yhat;
        $sum_error    += $error ** 2;
        $weights->[0]  = $weights->[0] + $l_rate * $error;
        for my $i (0 .. $#{$row}-1){
          $weights->[$i + 1] = $weights->[$i + 1] + $l_rate * $error * $row->[$i];
        }
      }
      printf " >epoch=%d, lrate=%.3f, error=%.3f\n", $epoch, $l_rate, $sum_error;
      
      # Storing performance data for the plot:
      push @train_loss, $sum_error;
      if (defined $args{test}){
        unless (ref($args{test}) eq 'ARRAY'){
          print STDERR "Test data is defined but it is not an array reference.\n";
        }else{
          $sum_error = 0.0;
          for my $row (@{$args{test}}){
            $yhat       = sml->predict($row, $weights);
            $error      = $row->[-1] - $yhat;
            $sum_error  += $error ** 2;
          }
          push @test_loss, $sum_error;
        }
      }
    }
    
    return $weights, \@train_loss, \@test_loss;
  }
  
  # Defined in Section 10.2.3 Sonar Case Study
  # Perceptron Algorithm With Stochastic Gradient Descent
  sub perceptron{
    my ($self, $train, $test, %args) = (splice(@_, 0, 3), 
                                        sml->get_arguments(l_rate  => undef, 
                                                           n_epoch => undef, \@_));
    my @predictions = ();
    my ($weights, $train_loss, $test_loss) = sml->train_weights($train, $args{l_rate}, $args{n_epoch}, $test);
    for my $row (@$test){
      my $prediction = sml->predict($row, $weights);
      push @predictions, sprintf '%.0f', $prediction;
    }
    return \@predictions, $train_loss, $test_loss;
  }

  # Defined in Section 11.2.1 Gini Index
  # Function To Calculate the Gini Index of a Dataset split.
  # Calculate the Gini index for a split dataset
  sub gini_index{
    my ($self, $groups, $classes) = @_;
  
    # count all samples at split point
    my $n_instances = sum(map { scalar @$_ } @$groups);
  
    # sum weighted Gini index for each group
    my ($gini, $size, $score) = 0.0;
    for my $group (@$groups) {
      $size = @$group;
          
      # avoid divide by zero
      next if $size == 0;
      
      $score = 0.0;
  
      # score the group based on the score for each class
      for my $class_val (@$classes){
        my $p = (grep { $_->[-1] == $class_val } @$group) / $size;
        $score += $p * $p;
      }
  
      # weight the group score by its relative size
      $gini += (1.0 - $score) * ($size / $n_instances);
    }
    
    return $gini;
  }
  
  # Defined in Section 11.2.2 Create Split
  # Function To Split a Dataset Based on a Split Point.
  # Split a dataset based on an attribute and an attribute value
  sub test_split{
    my ($self, $index, $value, $dataset) = @_;
    
    my (@left, @right);
    for my $row (@$dataset){
      if ($row->[$index] < $value){
        push @left, $row;
      }else{
        push @right, $row;
      }
    }
  
    return (\@left, \@right);
  }

  # Defined in Section 11.2.2 Create Split
  # Function To Find the Best Split Point in a Dataset.
  # Select the best split point for a dataset
  sub get_split{
    my ($self, $dataset) = @_;
  
    my ($b_index, $b_value, $b_score, $b_groups) = (999, 999, 999);
    my @class_values = grep {defined $_} my %unique = map { $_->[-1] => undef } @$dataset;
  
    for my $index (0 .. $#{$dataset->[0]} -1){
      for my $row (@$dataset) {
        my @groups = sml->test_split($index, $row->[$index], $dataset);
        my $gini   = sml->gini_index(\@groups, \@class_values);
        # printf "X%d < %.3f Gini=%.3f\n", ($index+1), $row->[$index], $gini;
        if ($gini < $b_score){
          ($b_index, $b_value, $b_score, $b_groups) = ($index, $row->[$index], $gini, \@groups);
        }
      }
    }
  
    return {index=>$b_index, value=>$b_value, groups=>$b_groups};
  }

  # Defined in Section 11.2.3 Build a Tree
  # Function To Create a Terminal Node.
  # Create a terminal node value
  sub to_terminal{
    my ($self, $group) = @_;
    my %outcomes = ();
    map { $outcomes{$_->[-1]}++ } @$group;
    return (sort { $outcomes{$b} <=> $outcomes{$a} } sort keys %outcomes)[0];
  }
  
  # Defined in Section 11.2.3 Build a Tree
  # Function To Create Split Points Recursively.
  # Create child splits for a node or make terminal
  sub split_node{
    my ($self, $node, $max_depth, $min_size, $depth) = @_;
    
    my ($left, $right) = @{$node->{'groups'}};
    delete $node->{'groups'};
  
    # check for a no split
    if (!@$left || !@$right){
      $node->{'left'} = $node->{'right'} = sml->to_terminal([@$left, @$right]);
      return;
    }
  
    # check for max depth
    if ($depth >= $max_depth){
      ($node->{'left'}, $node->{'right'})  = (sml->to_terminal($left), sml->to_terminal($right));
      return;
    }
  
    # process left child
    if (scalar(@$left) <= $min_size){
      $node->{'left'} = sml->to_terminal($left);
    }else{
      $node->{'left'} = sml->get_split($left);
      sml->split_node($node->{'left'}, $max_depth, $min_size, $depth + 1);
    }
  
    # process right child
    if (scalar(@$right) <= $min_size){
      $node->{'right'} = sml->to_terminal($right);
    }else{
      $node->{'right'} = sml->get_split($right);
      sml->split_node($node->{'right'}, $max_depth, $min_size, $depth + 1);
    }
  }

  # Defined in Section 11.2.3 Build a Tree
  # Function To Create a Decision Tree.
  # Build a decision tree
  sub build_tree{
    my ($self, $train, $max_depth, $min_size) = @_;
    my $root = sml->get_split($train);
    sml->split_node($root, $max_depth, $min_size, 1);
    return $root;
  }
  
  # Defined in Section 11.2.3 Build a Tree
  # Example of Creating a Decision Tree From the Contrived Dataset.
  # Example of building a tree
  sub print_tree{
    my ($self, $node, $depth) = @_;
    $depth //= 0;
      
    if (ref($node) eq 'HASH') {
      printf "%s[X%d < %.3f]\n", ' ' x $depth , ($node->{'index'} + 1), $node->{'value'};
      sml->print_tree($node->{'left'}, $depth + 1);
      sml->print_tree($node->{'right'}, $depth + 1);
    }else{
      printf "%s[%s]\n", ' ' x $depth , $node;
    }
  }
  
  # Defined in Section 11.2.5 Banknote Case Study
  # Classification and Regression Tree Algorithm
  sub decision_tree{
    my ($self, $train, $test, %args) = (splice(@_, 0, 3), sml->get_arguments(max_depth=>undef, 
                                                                             min_size=>undef, \@_));
  
    my $tree        = sml->build_tree($train, $args{max_depth}, $args{min_size});
    my $predictions = [map {sml->predict($tree, $_)} @$test];
  
    return $predictions, $tree;
  }

  # Defined in Section
  # Defined in Section
  # Defined in Section
  # Defined in Section
  # Defined in Section
  # Defined in Section
  # Defined in Section
  # Defined in Section
  # Defined in Section
  # Defined in Section
  
  #-----------eig-----------
  #Compute eigenvalues and right or left eigenvectors of a square matrix.
  #https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.eig.html
  #https://numpy.org/doc/stable/reference/generated/numpy.linalg.eig.html
  #The LEFT Eigenvectors are calculated the same way as the Right Eigenvectors, but with the Transposed matrix A.
  #See Appendix B.2, p. 424 - Eigenvalues and Eigenvectors from the doctoral thesis of PÃ©rez-Arriaga, "Selective modal analysis with applications to electric power systems" (1981).
  #https://dspace.mit.edu/bitstream/handle/1721.1/15875/08206193-MIT.pdf?sequence=2
  sub eig{
    my ($self, $A, %args) = (splice(@_, 0, 2), sml->get_arguments(left=>0, right=>1, \@_));
  
    #d2l->eig(A, left=>0, right=>1);
    
    if (!defined $A){
      print STDERR "TypeError: _eig_dispatcher() missing 1 required positional argument: 'A'.\n";
      return;
    }
    
    if (ref ($A) ne 'ARRAY'){
      print STDERR "TypeError: First parameter 'A' must be a reference to an ARRAY.\n";
      return;
    }
    
    $A = PDL::Core::pdl($A);
    
    my $ndim = $A->ndims;
    
    if ($ndim != 2){
      print STDERR "TypeError: $ndim-dimensional array given. Array must be a bidimensional.\n";
      return;
    }
    
    #Validation of optional parameters
    if ($args{right} !~ /^[01]$/){
      print STDERR "Right eigen vector must be a boolean: 0 (False) or 1 (True).\n";
      return;
    }

    if ($args{left} !~ /^[01]$/){
      print STDERR "Left eigen vector must be a boolean: 0 (False) or 1 (True).\n";
      return;
    }
    
    if ($args{right} == 0 && $args{left} == 0){
      print STDERR "TypeError: Both right and left eigen vectors are set to 0 (False). One of them must be 1 (True).\n";
      return;
    }

    my ($num_cols, $num_rows) = PDL::Core::list($A->shape);
    if ($num_rows != $num_cols){
      print STDERR "First parameter 'A' must be a square matrix. The given shape [$num_rows, $num_cols] does not comply.\n";
      return;
    }
    
    my $rvalues = PDL::Core::zeroes($num_rows);
    my $ivalues = PDL::Core::zeroes($num_rows);
    my $rvector = PDL::Core::zeroes($num_rows, $num_cols); # normalized right eigenvector
    my $lvector = PDL::Core::zeroes($num_rows, $num_cols); # normalized left eigenvector
    my $info = 0;
    BEGIN { require PDL::LinearAlgebra }
    PDL::LinearAlgebra::geev($A, $args{right}, $args{left}, $rvalues, $ivalues, $rvector, $lvector, $info);
    if ($info != 0){
      print STDERR "Error while calculating Eigen decomposition. Error code: $info.\n";
      return;
    }
    
    ($rvalues, $ivalues, $lvector, $rvector) = (PDL::Core::unpdl($rvalues), PDL::Core::unpdl($ivalues), PDL::Core::unpdl($lvector->transpose), PDL::Core::unpdl($rvector->transpose));
    
    if ($args{left} && $args{right} ) {
      return $rvalues, $ivalues, $lvector, $rvector;
    }elsif($args{left}){
      return $rvalues, $ivalues, $lvector;
    }elsif($args{right}){
      return $rvalues, $ivalues, $rvector;
    }
  }
  #----------\eig-----------
  
  sub embedplot{
    my ($self, $plot, %args) = (splice (@_, 0, 2), width=>800, height=>650, @_);
    
    if (`whereis sox` =~ m/wkhtmltoimage:\s*$/){
      print STDERR "wkhtmltoimage is not found in your system. Install wkhtmltoimage first before running this function.\n";
      return;
    }
    
    unless (ref ($plot) eq 'Chart::Plotly::Plot'){
      print STDERR "First parameter plot must be a Chart::Plotly::Plot.\n";
      return;
    }
    
    # Save to file
    my $html_path = 'plot.html';
    open my $fh, '>', $html_path or die "Cannot write plot.html: $!";
    print $fh $plot->html();
    close $fh;
  
    # Convert HTML to PNG using wkhtmltoimage
    my $png_path = "plot.png";
    my $cmd   = `wkhtmltoimage --quiet --width $args{width} --height $args{height} $html_path $png_path`; # "--width", "800", "--height", "600",
    print STDERR "Image generation failed: $cmd" if $cmd;
    # sleep 2;
    unlink $html_path;
  
    # Actually embed the graph into a Jupyter code cell
    IPerl->png( $png_path );
  }
  
  1;
}