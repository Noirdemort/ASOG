package asog;

import com.sun.tools.javac.util.Log;
import javax.swing.*;
import java.awt.*;
import java.awt.event.*;
import java.io.*;
import som;

import org.deeplearning4j.clustering.algorithm.Distance;
import org.deeplearning4j.clustering.kmeans.KMeansClustering;
import org.deeplearning4j.nn.conf.CacheMode;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.conf.layers.variational.BernoulliReconstructionDistribution;
import org.deeplearning4j.nn.conf.layers.variational.VariationalAutoencoder;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;

import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.learning.config.RmsProp;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;
import org.nd4j.linalg.primitives.Pair;
import org.nd4j.linalg.schedule.MapSchedule;
import org.nd4j.linalg.schedule.ScheduleType;

import java.util.HashMap;
import java.util.List;
import java.util.Map;



/**
 *  Manages the Neural Network Models
 *
 * {@code HIDDEN_LAYER_WIDTH 50}
 * {@code HIDDEN_LAYER_COUNT 2}
 * {@code seed Random}
 *
 *  @author Noirdemort
 *  @version 1.0
 *
 */
class ProcessingRelay extends JFrame{


    /**
     * RNN dimensions
     */
    private static final int HIDDEN_LAYER_WIDTH = 50;
    private static final int HIDDEN_LAYER_CONT = 2;
    private static final int seed = (int) (Math.random()*9999 + 1);


    /**
     * For training on mixed voice samples dataset and predicting voices count in a voice sample.
     *
     * @param features Array consisting of mfcc or spectogram features for a voice sample per row.
     * @param labels Each row indicates number of voices for a sample row.
     *
     * @return MultiLayerNetwork: Recurrent Neural Network with LSTM features.
     *
     *
     */
    MultiLayerNetwork rnn(double[] features, int[] labels) {

        INDArray featureVector = Nd4j.createFromArray(features);
        INDArray labelVector = Nd4j.createFromArray(labels);

        NeuralNetConfiguration.Builder builder = new NeuralNetConfiguration.Builder();
        builder.seed(123);
        builder.biasInit(0);
        builder.miniBatch(false);
        builder.updater(new RmsProp(0.001));
        builder.weightInit(WeightInit.RELU_UNIFORM);

        NeuralNetConfiguration.ListBuilder listBuilder = builder.list();

        // first difference, for rnns we need to use LSTM.Builder
        for (int i = 0; i < HIDDEN_LAYER_CONT; i++) {
            LSTM.Builder hiddenLayerBuilder = new LSTM.Builder();
            hiddenLayerBuilder.nIn(i == 0 ? featureVector.rank() : HIDDEN_LAYER_WIDTH);
            hiddenLayerBuilder.nOut(HIDDEN_LAYER_WIDTH);
            hiddenLayerBuilder.activation(Activation.TANH);
            listBuilder.layer(i, hiddenLayerBuilder.build());
        }

        // we need to use RnnOutputLayer for our RNN
        RnnOutputLayer.Builder outputLayerBuilder;
        outputLayerBuilder = new RnnOutputLayer.Builder(LossFunction.MEAN_SQUARED_LOGARITHMIC_ERROR);
        outputLayerBuilder.activation(Activation.SOFTMAX);
        outputLayerBuilder.nIn(HIDDEN_LAYER_WIDTH);
        outputLayerBuilder.nOut(labelVector.columns());
        listBuilder.layer(HIDDEN_LAYER_CONT, outputLayerBuilder.build());

        // create network
        MultiLayerConfiguration conf = listBuilder.build();
        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();
        net.setListeners(new ScoreIterationListener(1));
        net.setLearningRate(0.0001);
        net.setEpochCount(1000);
        net.setCacheMode(CacheMode.DEVICE);

        DataSet dataSet = new DataSet(featureVector, labelVector);
        SplitTestAndTrain sr = dataSet.splitTestAndTrain(0.8);

        net.fit(sr.getTrain());
        List<String> rx = net.predict(sr.getTest());

        System.out.println("F1 Score: " + net.f1Score(sr.getTest()));
        System.out.println("Summary: " + net.summary());

        double score = net.score(sr.getTest());
        System.out.println("Accuracy: " + score*100 + "%");
        net.computeGradientAndScore();
        Pair<Gradient, Double> grs = net.gradientAndScore();

        INDArray grad;
        grad = grs.getFirst().gradient();
        System.out.println("Computed Score: " + grs.getSecond());

        for(String r: rx){
            System.out.print(r+" ");
        }

        for (float x: grad.toFloatVector()) {
            System.out.print(x);
        }

        return net;
    }


    /**
     * For separating all distinct voices... well probably
     */
    void extractionPipeline(){

    }
    
    //Code to plot SOMs after training
    public void plot(int xDim, int yDim, int [] nodes)
	 {
		 // Determine the number of observations assigned to each node
		 int [] counts = new int[xDim * yDim];
		 for(int i = 0; i < nodes.length; i++)
		 {
			 counts[nodes[i]] += 1;
		 }
		 // Determine the maximum and minimum counts for shading
		 int maxCount = 0;
		 int minCount = Integer.MAX_VALUE;
		 for(int i = 0; i < counts.length; i++)
		 {
			if(counts[i] < minCount)
			{
				minCount = counts[i];
			}
			if(counts[i] > maxCount)
			{
				maxCount = counts[i];
			}
			//System.out.println("Node " + i + ": " + counts[i]);
		 }
		 // Create the window
		 JFrame map = new JFrame ("Kohonen network");
		 // Set the output resolution, don't let it exceed 1024x768
		 //double aspectRatio = (double)xdim / yDim;
		 String col = colorBox.getSelectedItem().toString();
		 map.setSize(800, 600);
		 map.setLayout (new GridLayout(yDim, xDim));
		 
		 // Plot the counts
		 JButton [] jB = new JButton[counts.length];
		 // Set the colors
		 float r = 0;
		 float g = 0;
		 float b = 0;
		 // Fill top to bottom, left to right
		 // instead of left to right, top to bttom
		 int index;
		 int columnCount = 0;
		 for(int i = 0; i < counts.length; i++)
		 {
			 // Calculuate the element to use since we're
			 // filling from top to bottom, left to right
			 index = (i / xDim) + columnCount * yDim;
			 // Get the RGB values
			 if(col.equals("Red"))
			 {
				 r = (float)counts[index] / maxCount;
			 }
			 else if(col.equals("Green"))
			 {
				 g = (float)counts[index] / maxCount;
			 }
			 else
			 {
				 b = (float)counts[index] / maxCount;
			 }
			 jB[index] = new JButton("");
			 jB[index].setBackground(new Color(r, g, b));
			 jB[index].setOpaque(true);
			 map.add(jB[index]);
			 columnCount++;
			 // Reset the column count if it grows too large
			 if(columnCount >= xDim)
			 {
				 columnCount = 0;
			 }
		 }
		 map.setVisible(true); 
	 }


    /**
     *  Self Organizing Maps for enhancing features
     */
    void som(double[][] audiofilesdata,int x, int y,int z)
    {
        int xVal;
        int yVal;
        int epochVal;
        try
        {
                xVal = x;
                yVal = y;
                epochVal = z;
                if(xVal <= 0 || yVal <= 0 || epochVal <= 0)
                {
                    throw new NumberFormatException();
                }
        }
        catch(NumberFormatException nfe)
        {
                return;
        }
        long startTime = System.nanoTime();
        SOM training = new SOM(audiofilesdata, xVal, yVal, epochVal);
        training.train();
		long endTime = System.nanoTime();
        plot(xVal, yVal, training.getNodes());
    }


    /**
     *  Auto-encoders with k-Means for extracting voices... in my head?
     *
     * @param features array rows consisting of mfcc features with mixed voices per row.
     * @param numClusters number of voices predicted by RNN model
     *
     * @return KMeansClustering : kmeans model with distinct voice clusters
     */
    KMeansClustering autoKencoders(double[] features, int numClusters){
        INDArray featureVector = Nd4j.createFromArray(features);

        Nd4j.getRandom().setSeed(seed);
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .updater(new RmsProp(1e-3))
                .weightInit(WeightInit.XAVIER)
                .l2(1e-4)
                .list()
                .layer(new VariationalAutoencoder.Builder()
                        .activation(Activation.LEAKYRELU)
                        .encoderLayerSizes(256, 256)        //2 encoder layers, each of size 256
                        .decoderLayerSizes(256, 256)        //2 decoder layers, each of size 256
                        .pzxActivationFunction(Activation.IDENTITY)  //p(z|data) activation function
                        .reconstructionDistribution(new BernoulliReconstructionDistribution(Activation.SIGMOID.getActivationFunction()))     //Bernoulli distribution for p(data|z) (binary or 0 to 1 data only)
                        .nIn(28 * 28)                       //Input size: 28x28
                        .nOut(2)                            //Size of the latent variable space: p(z|x). 2 dimensions here for plotting, use more in general
                        .build())
                .build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();
        net.setLearningRate(0.0001);
        net.setEpochCount(10);
        net.setCacheMode(CacheMode.DEVICE);

        //Get the variational autoencoder layer
        org.deeplearning4j.nn.layers.variational.VariationalAutoencoder vae
                = (org.deeplearning4j.nn.layers.variational.VariationalAutoencoder) net.getLayer(0);


//        INDArray latentSpaceValues = vae.activate(featureVector, false, LayerWorkspaceMgr.noWorkspaces());                     //Collect and record the latent space values before training starts
        vae.fit(featureVector, LayerWorkspaceMgr.noWorkspaces());

        // TODO("Implement Kmeans properly and verify auto-encoder model")
        final int MAX_ITERATIONS = 300;
        KMeansClustering kmeans = KMeansClustering.setup(numClusters, MAX_ITERATIONS, Distance.COSINE_DISTANCE);

        return kmeans;
    }


    /**
     * For voice fingerprint recognition... in case of proper voice distinction
     *
     * @param features Array consisting of mfcc or spectogram features of a single voice per row.
     * @param labels Each row indicates number of voices for a sample row.
     * @param outputNum Total types of output possible.
     *
     * @return MultiLayerNetwork: Convolutional Neural Network
     */
    MultiLayerNetwork cnn(double[] features, int[] labels, int outputNum){

        INDArray featureVector = Nd4j.createFromArray(features);
        INDArray labelVector = Nd4j.createFromArray(labels);

        // TODO("Verify height and weight and find value of output num to be used in above params")
        int height = featureVector.rows();
        int width = featureVector.columns();


        Map<Integer, Double> learningRateSchedule = new HashMap<>();
        learningRateSchedule.put(0, 0.06);
        learningRateSchedule.put(200, 0.05);
        learningRateSchedule.put(600, 0.028);
        learningRateSchedule.put(800, 0.0060);
        learningRateSchedule.put(1000, 0.001);


        int channels = 1;

        // TODO("Verify Dimensions in Config and validate InputType.convolutionalFlat dimensions")

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .l2(0.0005) // ridge regression value
                .updater(new Nesterovs(new MapSchedule(
                            ScheduleType.ITERATION,
                            learningRateSchedule))) // can also use new Adam(1e-3) in updater
                .weightInit(WeightInit.XAVIER)
                .list()
                .layer(new ConvolutionLayer.Builder(5, 5)
                        .nIn(channels)
                        .stride(1, 1)
                        .nOut(20)
                        .activation(Activation.IDENTITY)
                        .build())
                .layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2, 2)
                        .stride(2, 2)
                        .build())
                .layer(new ConvolutionLayer.Builder(5, 5)
                        .stride(1, 1) // nIn need not specified in later layers
                        .nOut(50)
                        .activation(Activation.IDENTITY)
                        .build())
                .layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2, 2)
                        .stride(2, 2)
                        .build())
                .layer(new DenseLayer.Builder().activation(Activation.RELU)
                        .nOut(500)
                        .build())
                .layer(new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(outputNum)
                        .activation(Activation.SOFTMAX)
                        .build())
                .setInputType(InputType.convolutionalFlat(height, width, channels))
                .build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();
        net.setListeners(new ScoreIterationListener(10));
        net.setLearningRate(0.0001);
        net.setEpochCount(10);
        net.setCacheMode(CacheMode.DEVICE);

        Log.format("Total num of params: {}", net.numParams());

        DataSet dataSet = new DataSet(featureVector, labelVector);
        SplitTestAndTrain sr = dataSet.splitTestAndTrain(0.8);

        net.fit(sr.getTrain());

        List<String> rx = net.predict(sr.getTest());

        System.out.println("F1 Score: " + net.f1Score(sr.getTest()));
        System.out.println("Summary: " + net.summary());

        double score = net.score(sr.getTest());
        System.out.println("Accuracy: " + score*100 + "%");
        net.computeGradientAndScore();
        Pair<Gradient, Double> grs = net.gradientAndScore();

        INDArray grad;
        grad = grs.getFirst().gradient();
        System.out.println("Computed Score: " + grs.getSecond());

        for(String r: rx){
            System.out.print(r+" ");
        }

        for (float x: grad.toFloatVector()) {
            System.out.print(x);
        }

        return net;

        /* For saving model to  disk */
//        File modelPath = new File(BASE_PATH + "/model_name.zip");
//        ModelSerializer.writeModel(net, modelPath, true);


    }

}
