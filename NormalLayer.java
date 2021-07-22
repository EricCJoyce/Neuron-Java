/**************************************************************************************************
 Neural Network library, by Eric C. Joyce

 A normalizing layer applies the four learned parameters to its input.
  m = learned mean
  s = learned standard deviation
  g = learned coefficient
  b = learned constant

 input vec{x}    output vec{y}
   [ x1 ]     [ g*((x1 - m)/s)+b ]
   [ x2 ]     [ g*((x2 - m)/s)+b ]
   [ x3 ]     [ g*((x3 - m)/s)+b ]
   [ x4 ]     [ g*((x4 - m)/s)+b ]
   [ x5 ]     [ g*((x5 - m)/s)+b ]

 Note that this file does NOT seed the randomizer. That should be done by the parent program.
***************************************************************************************************/

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.nio.charset.StandardCharsets;

public class NormalLayer
  {
    private int i;                                                  //  Length of the input vector (and output vector)
    private double m;                                               //  Mu: the mean learned during training
    private double s;                                               //  Sigma: the standard deviation learned during training
    private double g;                                               //  The factor learned during training
    private double b;                                               //  The constant learned during training

    private double out[];
    private String layerName;

    public NormalLayer(int numInputs, String nameStr)
      {
        int ctr;

        i = numInputs;
        m = 0.0;                                                    //  Initialize to no effect:
        s = 1.0;                                                    //  y = g * ((x - m) / s) + b
        g = 1.0;
        b = 0.0;
        out = new double[i];
        for(ctr = 0; ctr < i; ctr++)                                //  Blank out buffer
          out[ctr] = 0.0;
        layerName = nameStr;
      }

    public NormalLayer(int numInputs)
      {
        this(numInputs, "");
      }

    public void setM(double arg_m)
      {
        m = arg_m;
        return;
      }

    public void setS(double arg_s)
      {
        s = arg_s;
        return;
      }

    public void setG(double arg_g)
      {
        g = arg_g;
        return;
      }

    public void setB(double arg_b)
      {
        b = arg_b;
        return;
      }

    public void setName(String nameStr)
      {
        layerName = nameStr;
        return;
      }

    public void print()
      {
        System.out.printf("Input Length = %d\n", i);
        System.out.printf("Mean = %f\n", m);
        System.out.printf("Std.dev = %f\n", s);
        System.out.printf("Coefficient = %f\n", g);
        System.out.printf("Constant = %f\n", b);
        System.out.print("\n");

        return;
      }

    public int inputs()
      {
        return i;
      }

    public double mu()
      {
        return m;
      }

    public double sigma()
      {
        return s;
      }

    public double factor()
      {
        return g;
      }

    public double constant()
      {
        return b;
      }

    public String name()
      {
        return layerName;
      }

    public double[] output()
      {
        return out;
      }

    public int run(double[] x)
      {
        int j;
        for(j = 0; j < i; j++)
          out[j] = g * ((x[j] - m) / s) + b;
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
            System.out.println("ERROR: Unable to read number of Normalization Layer inputs.");
            return false;
          }
        try
          {
            m = fp.readDouble();                                    //  (double) Read mu from file
          }
        catch(IOException ioErr)
          {
            System.out.println("ERROR: Unable to read Normalization Layer attribute mu.");
            return false;
          }
        try
          {
            s = fp.readDouble();                                    //  (double) Read sigma from file
          }
        catch(IOException ioErr)
          {
            System.out.println("ERROR: Unable to read Normalization Layer attribute sigma.");
            return false;
          }
        try
          {
            g = fp.readDouble();                                    //  (double) Read factor from file
          }
        catch(IOException ioErr)
          {
            System.out.println("ERROR: Unable to read Normalization Layer factor.");
            return false;
          }
        try
          {
            b = fp.readDouble();                                    //  (double) Read constant from file
          }
        catch(IOException ioErr)
          {
            System.out.println("ERROR: Unable to read Normalization Layer constant.");
            return false;
          }

        buffer = new byte[NeuralNet.LAYER_NAME_LEN];
        for(ctr = 0; ctr < NeuralNet.LAYER_NAME_LEN; ctr++)         //  Blank out buffer
          buffer[ctr] = 0x00;
        for(ctr = 0; ctr < NeuralNet.LAYER_NAME_LEN; ctr++)
          {
            try
              {
                buffer[ctr] = fp.readByte();
              }
            catch(IOException ioErr)
              {
                System.out.println("ERROR: Unable to read Normalization Layer name.");
                return false;
              }
          }
        layerName = new String(buffer, StandardCharsets.UTF_8);     //  Convert byte array to String
        buffer = null;                                              //  Release the array
        System.gc();                                                //  Summon the garbage collector

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
            System.out.println("ERROR: Unable to write number of Normalization Layer inputs.");
            return false;
          }
        try
          {
            fp.writeDouble(m);                                      //  (double) Write mu to file
          }
        catch(IOException ioErr)
          {
            System.out.println("ERROR: Unable to write Normalization Layer attribute mu.");
            return false;
          }
        try
          {
            fp.writeDouble(s);                                      //  (double) Write sigma to file
          }
        catch(IOException ioErr)
          {
            System.out.println("ERROR: Unable to write Normalization Layer attribute sigma.");
            return false;
          }
        try
          {
            fp.writeDouble(g);                                      //  (double) Write factor to file
          }
        catch(IOException ioErr)
          {
            System.out.println("ERROR: Unable to write Normalization Layer factor.");
            return false;
          }
        try
          {
            fp.writeDouble(b);                                      //  (double) Write constant to file
          }
        catch(IOException ioErr)
          {
            System.out.println("ERROR: Unable to write Normalization Layer constant.");
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
                System.out.println("ERROR: Unable to write Normalization name to file.");
                return false;
              }
          }
        buffer = null;                                              //  Release the array
        System.gc();                                                //  Summon the garbage collector

        return true;
      }
  }
