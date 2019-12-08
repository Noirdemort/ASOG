package asog;

import java.net.MalformedURLException;
import java.net.URL;
import java.util.ArrayList;

import edu.cmu.sphinx.frontend.Data;
import edu.cmu.sphinx.frontend.DataProcessor;
import edu.cmu.sphinx.frontend.FrontEnd;
import edu.cmu.sphinx.frontend.frequencywarp.MelFrequencyFilterBank;
import edu.cmu.sphinx.frontend.transform.DiscreteCosineTransform2;
import edu.cmu.sphinx.frontend.transform.DiscreteFourierTransform;
import edu.cmu.sphinx.frontend.util.AudioFileDataSource;


public class FeatureGrid {

    Data mfcc(String filename, int minFreq, int maxFreq, int numFilters) throws MalformedURLException {
        // TODO("Figure out the meaning of these parameters");
        AudioFileDataSource audioDataSource = new AudioFileDataSource(3200, null);
        audioDataSource.setAudioFile(new URL(filename), "source");

        final ArrayList<DataProcessor> pipeline = new ArrayList<>();

        pipeline.add(audioDataSource.getPredecessor());
        pipeline.add(new DiscreteFourierTransform());
        pipeline.add(new MelFrequencyFilterBank(minFreq, maxFreq, numFilters));
        pipeline.add(new DiscreteCosineTransform2(numFilters, 12));
        FrontEnd f = new FrontEnd(pipeline);

        Data mfccs;
        do {
            mfccs = f.getData();
        } while(f.getData() != null);

        return mfccs;
    }

    void spectogram(){
        // TODO("write mel spectogram code here");
    }

}
