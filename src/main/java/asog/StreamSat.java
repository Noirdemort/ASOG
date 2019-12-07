package asog;

import java.io.*;
import java.nio.ByteBuffer;
import javax.sound.sampled.AudioFormat;
import javax.sound.sampled.AudioInputStream;
import javax.sound.sampled.AudioSystem;
import javax.sound.sampled.UnsupportedAudioFileException;

import javazoom.jl.converter.Converter;
import javazoom.jl.decoder.JavaLayerException;


class StreamSat {

    public enum AudioType { RAW, WAV, MP3 };

    protected AudioInputStream audio;
    protected AudioType audioType;
    protected byte[] audioData;


    AudioInputStream readWAV(String filename) throws IOException, UnsupportedAudioFileException{
        AudioInputStream ax = AudioSystem.getAudioInputStream(new File(filename));
        audio = ax;
        audioType = AudioType.WAV;
        setData();
        return ax;
    }


    AudioInputStream readMP3(String filename) throws IOException, UnsupportedAudioFileException, JavaLayerException {
        Converter conv = new Converter();
        conv.convert(filename, "temp.wav");
        File file = new File("temp.wav");
        AudioInputStream in= AudioSystem.getAudioInputStream(file);
        AudioInputStream din = null;
        AudioFormat baseFormat = in.getFormat();
        AudioFormat decodedFormat = new AudioFormat(AudioFormat.Encoding.PCM_SIGNED,
                baseFormat.getSampleRate(),
                16,
                baseFormat.getChannels(),
                baseFormat.getChannels() * 2,
                baseFormat.getSampleRate(),
                false);
        din = AudioSystem.getAudioInputStream(decodedFormat, in);
        audio = din;
        audioType = AudioType.MP3;
        setData();
        file.delete();
        return din;
    }


    AudioInputStream readRAW(InputStream file) throws IOException, UnsupportedAudioFileException{

        AudioInputStream ax = AudioSystem.getAudioInputStream(file);
        audio = ax;
        audioType = AudioType.RAW;
        setData();
        return ax;
    }


    private void setData() throws IOException{
        DataInputStream dis = new DataInputStream(audio);      //So we can use readFully()
        AudioFormat format = audio.getFormat();
        audioData = new byte[(int)(audio.getFrameLength() * format.getFrameSize())];
        dis.readFully(audioData);
        dis.close();
    }

    double[] toDoubleArray(){
        int times = Double.SIZE / Byte.SIZE;
        double[] doubles = new double[audioData.length / times];
        for(int i=0;i<doubles.length;i++){
            doubles[i] = ByteBuffer.wrap(audioData, i*times, times).getDouble();
        }
        return doubles;
    }


    void readDirectory(String directoryPath){

   }

}
