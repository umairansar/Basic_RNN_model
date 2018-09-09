# Basic_RNN_model
This model uses a mathematically created sign wave and tries to trace and learn the wave using RNN.

### Mathedology
The model was built using the Google's Tensorflow framework. The implementation can be broken down as follows:
      
      1- Create a sine wave function
      2- Pick a random section from the wave i.e. a batch
      3. Innitialize two variables with a difference of just one timestep.
      4- Apply RNN function and optimizer
      5- Plot a graph comparing actual vs predicted sequence (or wave)


### Challenges
1. Take serious note of the variables used. While training y0 is a batch of time series and y1 is the expected output one time step ahead.  

2. If you run this in colab or jupyter notebooks repeatedy you might get dimension error or likewise. 

### Resolution
1. Use a loop to compute a whole wave using new predicted value of time series i.e. a wave composed of all one-time-step-ahead predicted values.

2. Add this before running the RNN Cell and the placeholders used in it: tf.reset_default_graph() 

### Parameter used

num_inputs = 1
num_neurons = 100
num_outputs = 1
learning_rate = 0.001
epochs = 2000
batch_size = 1

### Note
Here the dataset was the sine wave itself. Unlike CNN, this doesnt use a conventional type of dataset. The accuracy is determined visually by checking how much trace is generated to overlap successfully on the original wave. Please check out the code to view these concepts fully.
