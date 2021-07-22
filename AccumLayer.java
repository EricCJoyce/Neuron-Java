/**************************************************************************************************
 Neural Network library, by Eric C. Joyce

 Accumulation Layers are collection tools, pass-throughs

 Note that this file does NOT seed the randomizer. That should be done by the parent program.
***************************************************************************************************/

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.nio.charset.StandardCharsets;

public class AccumLayer
  {
    private int i;                                                  //  Number of inputs--ACCUMULATORS GET NO bias-1
    private String layerName;
    private double out[];

    /* Accumulator Layers have as many outputs as inputs. Additionally set the name. */
    public AccumLayer(int inputs, String nameStr)
      {
        i = inputs;
        out = new double[i];
        layerName = nameStr;
      }

    /* Accumulator Layers have as many outputs as inputs. */
    public AccumLayer(int inputs)
      {
        this(inputs, "");
      }

    public void set(int index, double val)
      {
        if(index < i)
          out[index] = val;
        return;
      }

    /* Set the name of the Accumulator Layer */
    public void setName(String nameStr)
      {
        layerName = nameStr;
        return;
      }

    public int inputs()
      {
        return i;
      }

    public double[] output()
      {
        return out;
      }

    public String name()
      {
        return layerName;
      }

    /* Return the layer's output length */
    public int outputLen()
      {
        return i;
      }

    public boolean read(DataInputStream fp)
      {
        int ctr;
        byte buffer[];

        try
          {
            i = fp.readInt();                                       //  (int) Read number of layer inputs from file
          }
        catch(IOException ioErr)
          {
            System.out.println("ERROR: Unable to read number of Accumulation Layer inputs.");
            return false;
          }

        out = new double[i];                                        //  (Re)Allocate output buffer

        buffer = new byte[NeuralNet.LAYER_NAME_LEN];
        for(ctr = 0; ctr < NeuralNet.LAYER_NAME_LEN; ctr++)
          {
            try
              {
                buffer[ctr] = fp.readByte();
              }
            catch(IOException ioErr)
              {
                System.out.println("ERROR: Unable to read Accumulation Layer name.");
                return false;
              }
          }
        layerName = new String(buffer, StandardCharsets.UTF_8);     //  Convert byte array to String

        buffer = null;                                              //  Release the array
        System.gc();                                                //  Call the garbage collector

        return true;
      }

    public boolean write(DataOutputStream fp)
      {
        int ctr;
        byte buffer[];

        try
          {
            fp.writeInt(i);                                         //  (int) Write number of layer inputs to file
          }
        catch(IOException ioErr)
          {
            System.out.println("ERROR: Unable to write number of Accumulation Layer inputs.");
            return false;
          }

        buffer = new byte[NeuralNet.LAYER_NAME_LEN];                //  Allocate
        for(ctr = 0; ctr < NeuralNet.LAYER_NAME_LEN; ctr++)         //  Blank out buffer
          buffer[ctr] = 0x00;
        buffer = layerName.getBytes(StandardCharsets.UTF_8);        //  Write layer name to file
        for(ctr = 0; ctr < NeuralNet.LAYER_NAME_LEN; ctr++)         //  Blank out buffer
          {
            try
              {
                fp.write(buffer[ctr]);
              }
            catch(IOException ioErr)
              {
                System.out.println("ERROR: Unable to write Accumulation Layer name to file.");
                return false;
              }
          }

        buffer = null;                                              //  Release the array
        System.gc();                                                //  Call the garbage collector

        return true;
      }
  }
