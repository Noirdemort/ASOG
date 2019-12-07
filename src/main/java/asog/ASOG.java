package asog;
import org.tensorflow.Graph;

import java.nio.ByteBuffer;

import javax.sound.sampled.AudioFormat;
import javax.sound.sampled.AudioInputStream;

// Advanced Sonic Operations Grid

public class ASOG {
    public static void main(String[] args) {
        System.out.println("Hello Arbys");

        StreamSat streamSat = new StreamSat();
        try {
            AudioInputStream ax = streamSat.readMP3("/Users/noirdemort/chernobyl.mp3");
        } catch (Exception e){
            e.printStackTrace();
        }
        System.out.println(streamSat.audioData);

        double []axr = streamSat.toDoubleArray();

        for(byte i:streamSat.audioData)
            System.out.print(i+" ");


    }
}
