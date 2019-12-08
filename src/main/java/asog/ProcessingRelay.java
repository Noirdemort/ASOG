package asog;

import org.deeplearning4j.nn.conf.CacheMode;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.RmsProp;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.primitives.Pair;
import java.util.List;



/**
 *  Manages the Neural Network Model
 *
 *  @author Noirdemort
 */
public class ProcessingRelay {


    /**
     * RNN dimensions
     */
    private static final int HIDDEN_LAYER_WIDTH = 50;
    private static final int HIDDEN_LAYER_CONT = 2;


    /**
     * For training on known mixed voice samples dataset
     * and predicting number of voices in a voice sample
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
        RnnOutputLayer.Builder outputLayerBuilder = new RnnOutputLayer.Builder(LossFunctions.LossFunction.MEAN_SQUARED_LOGARITHMIC_ERROR);
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

        DataSet trainingData = new DataSet(featureVector, labelVector);
        SplitTestAndTrain sr = trainingData.splitTestAndTrain(0.2);

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

    /**
     *  Self Organizing Maps for enhancing features
     */
    void som(){

    }

    /**
     *  Auto-encoders with k-Means for extracting voices
     */
    void  autoKencoders(){

    }


    /**
     * For Recognizing each voice and voice fingerprint recognition... in case of proper voice distinction
     */
    void cnn(){

    }

}
