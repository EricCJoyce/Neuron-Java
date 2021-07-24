/*
  Eric C. Joyce, Stevens Institute of Technology, 2020

  jblas-1.2.4.jar needs to be in the same directory as all the Neuron *.java files and BuildNeuronModel.java.

  This program creates a network file named "mnist.nn".
  Use the make file to compile.

  Run by calling:
    java -cp .:./jblas-1.2.4.jar BuildNeuronModel
*/

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

public class BuildNeuronModel
  {
    public static void main(String[] args)
      {
        NeuralNet nn;
        String filename;
        double w[];
        int i, j;

        DataInputStream fp;
        ByteBuffer byteBuffer;
        byte byteArr[];

        nn = new NeuralNet(784);                                    //  Initialize for 28 * 28 images, flattened

        /**************************************************************************/
        /***********************************************************    C O N V 2 */
        nn.addConv2D(28, 28);                                       //  Add a Conv2D layer that receives the input: 28 x 28
        nn.conv(0).setName("Conv2D-1");                             //  Name the Conv2D layer
        //////////////////////////////////////////////////////////////  Add 8 (3 x 3) kernels, each = 10 weights
        nn.conv(0).addFilter(3, 3);                                 //  filter[0][0]
        nn.conv(0).addFilter(3, 3);                                 //  filter[0][1]
        nn.conv(0).addFilter(3, 3);                                 //  filter[0][2]
        nn.conv(0).addFilter(3, 3);                                 //  filter[0][3]
        nn.conv(0).addFilter(3, 3);                                 //  filter[0][4]
        nn.conv(0).addFilter(3, 3);                                 //  filter[0][5]
        nn.conv(0).addFilter(3, 3);                                 //  filter[0][6]
        nn.conv(0).addFilter(3, 3);                                 //  filter[0][7]

        for(i = 0; i < 8; i++)
          {
            filename = new String(String.format("%s/Conv2D-%d.weights", args[0], i));
            try
              {
                fp = new DataInputStream(new FileInputStream(filename));
              }
            catch(FileNotFoundException fileErr)
              {
                System.out.printf("ERROR: Unable to open %s.\n", filename);
                return;
              }

            byteArr = new byte[80];                                 //  Allocate space for 10 doubles (8 bytes each)

            try
              {
                fp.read(byteArr);
              }
            catch(IOException ioErr)
              {
                System.out.println("ERROR: Unable to read weights from file.");
                return;
              }

            byteBuffer = ByteBuffer.allocate(80);
            byteBuffer = ByteBuffer.wrap(byteArr);
            byteBuffer.order(ByteOrder.LITTLE_ENDIAN);              //  Read little-endian

            w = new double[10];                                     //  Allocate weights array

            for(j = 0; j < 10; j++)
              w[j] = byteBuffer.getDouble();

            nn.conv(0).setW_i(w, i);                                //  Set weights for filter[0][i]

            try
              {
                fp.close();
              }
            catch(IOException ioErr)
              {
                System.out.println("ERROR: Unable to close file.");
                return;
              }
          }

        //////////////////////////////////////////////////////////////  Add 16 (5 x 5) kernels, each = 26 weights
        nn.conv(0).addFilter(5, 5);                                 //  filter[0][8]
        nn.conv(0).addFilter(5, 5);                                 //  filter[0][9]
        nn.conv(0).addFilter(5, 5);                                 //  filter[0][10]
        nn.conv(0).addFilter(5, 5);                                 //  filter[0][11]
        nn.conv(0).addFilter(5, 5);                                 //  filter[0][12]
        nn.conv(0).addFilter(5, 5);                                 //  filter[0][13]
        nn.conv(0).addFilter(5, 5);                                 //  filter[0][14]
        nn.conv(0).addFilter(5, 5);                                 //  filter[0][15]
        nn.conv(0).addFilter(5, 5);                                 //  filter[0][16]
        nn.conv(0).addFilter(5, 5);                                 //  filter[0][17]
        nn.conv(0).addFilter(5, 5);                                 //  filter[0][18]
        nn.conv(0).addFilter(5, 5);                                 //  filter[0][19]
        nn.conv(0).addFilter(5, 5);                                 //  filter[0][20]
        nn.conv(0).addFilter(5, 5);                                 //  filter[0][21]
        nn.conv(0).addFilter(5, 5);                                 //  filter[0][22]
        nn.conv(0).addFilter(5, 5);                                 //  filter[0][23]

        for(i = 8; i < 24; i++)
          {
            filename = new String(String.format("%s/Conv2D-%d.weights", args[0], i));
            try
              {
                fp = new DataInputStream(new FileInputStream(filename));
              }
            catch(FileNotFoundException fileErr)
              {
                System.out.printf("ERROR: Unable to open %s.\n", filename);
                return;
              }

            byteArr = new byte[208];                                //  Allocate space for 26 doubles (8 bytes each)

            try
              {
                fp.read(byteArr);
              }
            catch(IOException ioErr)
              {
                System.out.println("ERROR: Unable to read weights from file.");
                return;
              }

            byteBuffer = ByteBuffer.allocate(208);
            byteBuffer = ByteBuffer.wrap(byteArr);
            byteBuffer.order(ByteOrder.LITTLE_ENDIAN);              //  Read little-endian

            w = new double[26];                                     //  Allocate weights array

            for(j = 0; j < 26; j++)
              w[j] = byteBuffer.getDouble();

            nn.conv(0).setW_i(w, i);                                        //  Set weights for filter[0][i]

            try
              {
                fp.close();
              }
            catch(IOException ioErr)
              {
                System.out.println("ERROR: Unable to close file.");
                return;
              }
          }

        /**************************************************************************/
        /***********************************************************    A C C U M */
        nn.addAccum(14624);                                         //  Add accumulator layer (ACCUM_ARRAY, 0)
        nn.accum(0).setName("Accum-1");                             //  Name the first accumulator layer

        /**************************************************************************/
        /***********************************************************    D E N S E */
        nn.addDense(14624, 100);                                    //  Add dense layer (DENSE_ARRAY, 0)
        nn.dense(0).setName("Dense-0");                             //  Name the first dense layer
        filename = new String(String.format("%s/Dense-0.weights", args[0]));
        try
          {
            fp = new DataInputStream(new FileInputStream(filename));
          }
        catch(FileNotFoundException fileErr)
          {
            System.out.printf("ERROR: Unable to open %s.\n", filename);
            return;
          }

        byteArr = new byte[11700000];                               //  Allocate space for dense layer doubles (8 bytes each)

        try
          {
            fp.read(byteArr);
          }
        catch(IOException ioErr)
          {
            System.out.println("ERROR: Unable to read weights from file.");
            return;
          }

        byteBuffer = ByteBuffer.allocate(11700000);
        byteBuffer = ByteBuffer.wrap(byteArr);
        byteBuffer.order(ByteOrder.LITTLE_ENDIAN);                  //  Read little-endian

        w = new double[1462500];                                    //  Allocate weights array

        for(j = 0; j < 1462500; j++)
          w[j] = byteBuffer.getDouble();

        nn.dense(0).setW(w);                                        //  Set weights for first dense layer

        try
          {
            fp.close();
          }
        catch(IOException ioErr)
          {
            System.out.println("ERROR: Unable to close file.");
            return;
          }

        //////////////////////////////////////////////////////////////  Add second dense layer
        nn.addDense(100, 10);                                       //  Add dense layer (DENSE_ARRAY, 1)
        nn.dense(1).setName("Dense-1");                             //  Name the second dense layer
        filename = new String(String.format("%s/Dense-1.weights", args[0]));
        try
          {
            fp = new DataInputStream(new FileInputStream(filename));
          }
        catch(FileNotFoundException fileErr)
          {
            System.out.printf("ERROR: Unable to open %s.\n", filename);
            return;
          }

        byteArr = new byte[8080];                                   //  Allocate space for dense layer doubles (8 bytes each)

        try
          {
            fp.read(byteArr);
          }
        catch(IOException ioErr)
          {
            System.out.println("ERROR: Unable to read weights from file.");
            return;
          }

        byteBuffer = ByteBuffer.allocate(8080);
        byteBuffer = ByteBuffer.wrap(byteArr);
        byteBuffer.order(ByteOrder.LITTLE_ENDIAN);                  //  Read little-endian

        w = new double[1010];                                       //  Allocate weights array

        for(j = 0; j < 1010; j++)
          w[j] = byteBuffer.getDouble();

        nn.dense(1).setW(w);                                        //  Set weights for second dense layer

        try
          {
            fp.close();
          }
        catch(IOException ioErr)
          {
            System.out.println("ERROR: Unable to close file.");
            return;
          }

        nn.dense(1).setF_i(ActivationFunction.SOFTMAX, 0);          //  Set output layer's activation function to softmax
        nn.dense(1).setF_i(ActivationFunction.SOFTMAX, 1);          //  (Because we allow units to have different activation functions
        nn.dense(1).setF_i(ActivationFunction.SOFTMAX, 2);          //  you have to set SOFTMAX for all output nodes.)
        nn.dense(1).setF_i(ActivationFunction.SOFTMAX, 3);
        nn.dense(1).setF_i(ActivationFunction.SOFTMAX, 4);
        nn.dense(1).setF_i(ActivationFunction.SOFTMAX, 5);
        nn.dense(1).setF_i(ActivationFunction.SOFTMAX, 6);
        nn.dense(1).setF_i(ActivationFunction.SOFTMAX, 7);
        nn.dense(1).setF_i(ActivationFunction.SOFTMAX, 8);
        nn.dense(1).setF_i(ActivationFunction.SOFTMAX, 9);

        /******************************************************************************/
        /******************************************************************************/
        /******************************************************************************/

                                                                    //  Connect input to conv2d[0]
        if(!nn.linkLayers(NeuralNet.INPUT_ARRAY, 0, 0, 784, NeuralNet.CONV2D_ARRAY, 0))
          System.out.println(">>>                Link[0] failed");
                                                                    //  Connect conv2d[0] to accum[0]
        if(!nn.linkLayers(NeuralNet.CONV2D_ARRAY, 0, 0, 14624, NeuralNet.ACCUM_ARRAY, 0))
          System.out.println(">>>                Link[1] failed");
                                                                    //  Connect accum[0] to dense[0]
        if(!nn.linkLayers(NeuralNet.ACCUM_ARRAY, 0, 0, 14624, NeuralNet.DENSE_ARRAY, 0))
          System.out.println(">>>                Link[2] failed");
                                                                    //  Connect dense[0] to dense[1]
        if(!nn.linkLayers(NeuralNet.DENSE_ARRAY, 0, 0, 100, NeuralNet.DENSE_ARRAY, 1))
          System.out.println(">>>                Link[3] failed");

        nn.sortEdges();
        nn.printEdgeList();
        System.out.print("\n\n");
        nn.print();

        nn.write("mnist.nn");

        return;
      }
  }