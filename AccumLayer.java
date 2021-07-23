/**************************************************************************************************
 Neural Network library, by Eric C. Joyce

 Accumulation Layers are collection tools, pass-throughs

 Note that this file does NOT seed the randomizer. That should be done by the parent program.
***************************************************************************************************/

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
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

    /* Use placeholder arguments */
    public AccumLayer()
      {
        this(1, "");
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

        ByteBuffer byteBuffer;
        int allocation;
        byte byteArr[];

        allocation = 4 + NeuralNet.LAYER_NAME_LEN;                  //  Allocate space for 1 int and the layer name
        byteArr = new byte[allocation];

        try
          {
            fp.read(byteArr);
          }
        catch(IOException ioErr)
          {
            System.out.println("ERROR: Unable to read Accumulation Layer from file.");
            return false;
          }

        byteBuffer = ByteBuffer.allocate(allocation);
        byteBuffer = ByteBuffer.wrap(byteArr);
        byteBuffer.order(ByteOrder.LITTLE_ENDIAN);                  //  Read little-endian

        i = byteBuffer.getInt();                                    //  (int) Read the number of inputs from file

        byteArr = new byte[NeuralNet.LAYER_NAME_LEN];               //  Allocate
        for(ctr = 0; ctr < NeuralNet.LAYER_NAME_LEN; ctr++)         //  Read into array
          byteArr[ctr] = byteBuffer.get();
        layerName = new String(byteArr, StandardCharsets.UTF_8);

        out = new double[i];                                        //  (Re)Allocate output buffer

        byteArr = null;                                             //  Release the array
        System.gc();                                                //  Call the garbage collector

        return true;
      }

    public boolean write(DataOutputStream fp)
      {
        int ctr;

        ByteBuffer byteBuffer;
        int allocation;
        byte byteArr[];
                                                                    //  Allocate space for
        allocation = 4 + NeuralNet.LAYER_NAME_LEN;                  //  1 int and the layer name.
        byteBuffer = ByteBuffer.allocate(allocation);
        byteBuffer.order(ByteOrder.LITTLE_ENDIAN);                  //  Write little-endian

        byteBuffer.putInt(i);                                       //  (int) Save AccumLayer input size to file

        byteArr = new byte[NeuralNet.LAYER_NAME_LEN];               //  Allocate
        for(ctr = 0; ctr < NeuralNet.LAYER_NAME_LEN; ctr++)         //  Blank out buffer
          byteArr[ctr] = 0x00;
        ctr = 0;                                                    //  Fill in up to limit
        while(ctr < NeuralNet.LAYER_NAME_LEN && ctr < layerName.length())
          {
            byteArr[ctr] = (byte)layerName.codePointAt(ctr);
            ctr++;
          }
        for(ctr = 0; ctr < NeuralNet.LAYER_NAME_LEN; ctr++)         //  Write layer name to file
          byteBuffer.put(byteArr[ctr]);

        byteArr = byteBuffer.array();

        try
          {
            fp.write(byteArr, 0, byteArr.length);
          }
        catch(IOException ioErr)
          {
            System.out.println("ERROR: Unable to write Accumulation Layer to file.");
            return false;
          }

        byteArr = null;                                             //  Release the array
        System.gc();                                                //  Call the garbage collector

        return true;
      }
  }
