package asog;

import java.io.*;
import java.nio.ByteBuffer;
import javax.sound.sampled.AudioFormat;
import javax.sound.sampled.AudioInputStream;
import javax.sound.sampled.AudioSystem;
import javax.sound.sampled.UnsupportedAudioFileException;

import javazoom.jl.converter.Converter;
import javazoom.jl.decoder.JavaLayerException;


/**
 * Used for reading files in different formats: {@code RAW, WAV, MP3}
 *
 * @author noirdemort
 * @version 1.0
 */
class StreamSat {

    public enum AudioType { RAW, WAV, MP3 }

    /**
     * Inner Audio Features.
     *
     * audio: AudioStream type, is set when a read[AudioType] method is used.
     * audioType: Indicates the file format used when the data was read.
     * audioData: byte array representation of audio data from file.
     *
     */
    AudioInputStream audio;
    AudioType audioType;
    byte[] audioData;


    /**
     * Reads .wav file and converts to AudioInputStream and sets byte array and file format
     *
     * @param filename Filename consisting of full file path
     * @return AudioInputStream for file
     * @throws IOException Raised in case file is not available or accessible
     * @throws UnsupportedAudioFileException Raised in case of unsupported formats by Package.
     */
    AudioInputStream readWAV(String filename) throws IOException, UnsupportedAudioFileException{
        AudioInputStream ax = AudioSystem.getAudioInputStream(new File(filename));
        audio = ax;
        audioType = AudioType.WAV;
        setData();
        return ax;
    }


    /**
     * Reads .mp3 file and converts to AudioInputStream and sets byte array and file format
     *
     * @param filename Filename consisting of full file path
     * @return AudioInputStream for file
     *
     * @throws IOException Raised in case file is not available or accessible
     * @throws UnsupportedAudioFileException Raised in case of unsupported formats by Package
     * @throws JavaLayerException Raised in case conversion of MP3 to .wav fails
     */
    AudioInputStream readMP3(String filename) throws IOException, UnsupportedAudioFileException, JavaLayerException {
        Converter conv = new Converter();
        conv.convert(filename, "temp.wav");
        File file = new File("temp.wav");
        AudioInputStream in= AudioSystem.getAudioInputStream(file);
        AudioInputStream din;
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
        final boolean delete = file.delete();
        return din;
    }

    /**
     * Reads raw data and converts to AudioInputStream and sets byte array and file format
     *
     * @param file file InputStream consisting of raw audio data
     * @return AudioInputStream for file
     * @throws IOException Raised in case InputStream is not available or accessible
     * @throws UnsupportedAudioFileException Raised in case of unsupported formats by Package.
     */
    AudioInputStream readRAW(InputStream file) throws IOException, UnsupportedAudioFileException{

        AudioInputStream ax = AudioSystem.getAudioInputStream(file);
        audio = ax;
        audioType = AudioType.RAW;
        setData();
        return ax;
    }

    /**
     * Sets byte array and file format, is called automatically by other read methods.
     *
     * @throws IOException Raised in case file is not available or accessible
     */
    private void setData() throws IOException{
        DataInputStream dis = new DataInputStream(audio);      //So we can use readFully()
        AudioFormat format = audio.getFormat();
        audioData = new byte[(int)(audio.getFrameLength() * format.getFrameSize())];
        dis.readFully(audioData);
        dis.close();
    }

    /**
     * Convert byte array audio data to double array format.
     *
     *
     * @return Double Array consisting of audio data
     */
    double[] toDoubleArray(){
        int times = Double.SIZE / Byte.SIZE;
        double[] doubles = new double[audioData.length / times];
        for(int i=0;i<doubles.length;i++){
            doubles[i] = ByteBuffer.wrap(audioData, i*times, times).getDouble();
        }
        return doubles;
    }

    /**
     * Reads Audio and stores audio files recursively from a given path.
     *
     * @param directoryPath path from which the audio files are to be read recursively.
     */
    double[][] readDirectory(String directoryPath){
        File directory=new File(directoryPath);
        File[] contents=directory.listFiles();
        String fileSeparator = System.getProperty("file.separator");
        String absoluteFilePath = fileSeparator+"Users"+fileSeparator+"AudioFileArrays"+fileSeparator+"audiofilesdata.txt";
        File file = new File(absoluteFilePath);
        file = new File("audiofilesdata.txt");
        int l=directory.listFiles().length;
        int j=0;
        
        double[][] filearrays=new double[][l];
        
        for( File f:contents)
        {
            String extension = getFileExtension(new File(f.getAbsolutePath()));
            if(extension == "*.wav")
            {
                AudioInputStream as=readWAV(f.getName());
                double[] d=toDoubleArray();
                filearrays[j]=d;
                j++;
                BufferedWriter outputWriter = null;
                  outputWriter = new BufferedWriter(new FileWriter(file.getName()));
                  for (int i = 0; i < d.length; i++) {
                    
                    outputWriter.write(d[i]+"");
                    
                    outputWriter.write(Double.toString(d[i]);
                    outputWriter.newLine();
                  }
                  outputWriter.flush();  
                  outputWriter.close();  
            }
            else if(extension== "*.mp3")
            {
                AudioInputStream as=readMP3(f.getName());
                double[] d=toDoubleArray();
                filearrays[j]=d;
                j++;
                BufferedWriter outputWriter = null;
                  outputWriter = new BufferedWriter(new FileWriter(file.getName()));
                  for (int i = 0; i < d.length; i++) {
                    
                    outputWriter.write(d[i]+"");
                    
                    outputWriter.write(Double.toString(d[i]);
                    outputWriter.newLine();
                  }
                  outputWriter.flush();  
                  outputWriter.close(); 
            }
            else 
            {
                 InputStream inputStream = new FileInputStream(f.getAbsolutePath());
                 AudioInputStream as= readRAW(inputStream);
                 double[] d=toDoubleArray();
                 filearrays[j]=d;
                 j++;
                 BufferedWriter outputWriter = null;
                  outputWriter = new BufferedWriter(new FileWriter(file.getName()));
                  for (int i = 0; i < d.length; i++) {
                    
                    outputWriter.write(d[i]+"");
                    
                    outputWriter.write(Double.toString(d[i]);
                    outputWriter.newLine();
                  }
                  outputWriter.flush();  
                  outputWriter.close(); 
            }
        }
     return filearrays;
}
