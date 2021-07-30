/**************************************************************************************************
 Neural Network library, by Eric C. Joyce

 Model a GRU Layer as matrices and vectors:
  d = the length of a single input instance
      (that is, we may have an indefinitely long sequence of word-vectors, but each is an input instance of length 'd')
  h = the length of internal state vectors
  cache = the number of previous states to track and store

  input d-vec{x}       weights Wz              weights Wr              weights Wh
 [ x1 x2 x3 x4 ]        (h by d)                (h by d)                (h by d)
                 [ wz11 wz12 wz13 wz14 ] [ wr11 wr12 wr13 wr14 ] [ wh11 wh12 wh13 wh14 ]
                 [ wz21 wz22 wz23 wz24 ] [ wr21 wr22 wr23 wr24 ] [ wh21 wh22 wh23 wh24 ]
                 [ wz31 wz32 wz33 wz34 ] [ wr31 wr32 wr33 wr34 ] [ wh31 wh32 wh33 wh34 ]

                       weights Uz              weights Ur              weights Uh
                        (h by h)                (h by h)                (h by h)
                 [ uz11 uz12 uz13 ]      [ ur11 ur12 ur13 ]      [ uh11 uh12 uh13 ]
                 [ uz21 uz22 uz23 ]      [ ur21 ur22 ur23 ]      [ uh21 uh22 uh23 ]
                 [ uz31 uz32 uz33 ]      [ ur31 ur32 ur33 ]      [ uh31 uh32 uh33 ]

                     bias h-vec{bz}          bias h-vec{br}          bias h-vec{bh}
                 [ bz1 ]                 [ br1 ]                 [ bh1 ]
                 [ bz2 ]                 [ br2 ]                 [ bh2 ]
                 [ bz3 ]                 [ br3 ]                 [ bh3 ]

         H state cache (times 1, 2, 3, 4 = columns 0, 1, 2, 3)
        (h by cache)
 [ H11 H12 H13 H14 ]
 [ H21 H22 H23 H24 ]
 [ H31 H32 H33 H34 ]

 Note that this file does NOT seed the randomizer. That should be done by the parent program.
***************************************************************************************************/

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import org.jblas.*;
import static org.jblas.DoubleMatrix.*;

public class GRULayer
  {
    private int d;                                                  //  Dimensionality of input vector
    private int h;                                                  //  Dimensionality of hidden state vector
    private int cache;                                              //  The number of states to keep in memory:
                                                                    //  when 't' exceeds this, shift out.
    private int t;                                                  //  The time step
                                                                    //  W matrices are (h by d)
    private DoubleMatrix Wz;                                        //  Updtae gate weights
    private DoubleMatrix Wr;                                        //  Reset gate weights
    private DoubleMatrix Wh;                                        //  Output gate weights
                                                                    //  U matrices are (h by h)
    private DoubleMatrix Uz;                                        //  Recurrent connection update gate weights
    private DoubleMatrix Ur;                                        //  Recurrent connection reset gate weights
    private DoubleMatrix Uh;                                        //  Recurrent connection output gate weights
                                                                    //  Bias vectors are length h
    private DoubleMatrix bz;                                        //  Update gate bias
    private DoubleMatrix br;                                        //  Reset gate bias
    private DoubleMatrix bh;                                        //  Output gate bias

    private DoubleMatrix H;                                         //  Hidden state cache matrix (h by cache)
    private String layerName;

    /*  */
    public GRULayer(int dimInput, int dimState, int cacheSize, String nameStr)
      {
        int x, y;
        double xavier;

        d = dimInput;                                               //  Save dimensionality of input
        h = dimState;                                               //  Save dimensionality of states
        cache = cacheSize;                                          //  Save the cache size
        t = 0;                                                      //  Initial time step = 0

        Wz = new DoubleMatrix(dimState, dimInput);                  //  Allocate this layer's W matrices
        Wr = new DoubleMatrix(dimState, dimInput);
        Wh = new DoubleMatrix(dimState, dimInput);

        Uz = new DoubleMatrix(dimState, dimState);                  //  Allocate this layer's U matrices
        Ur = new DoubleMatrix(dimState, dimState);
        Uh = new DoubleMatrix(dimState, dimState);

        bz = new DoubleMatrix(dimState);                            //  Allocate this newest layer's b vectors
        br = new DoubleMatrix(dimState);
        bh = new DoubleMatrix(dimState);

        H = new DoubleMatrix(dimState, cacheSize);                  //  Allocate this newest layer's cache matrix

        xavier = Math.sqrt(6 / (dimInput + dimState));
        for(y = 0; y < h; y++)                                      //  Xavier-initialize W matrices
          {
            for(x = 0; x < d; x++)
              {
                                                                    //  Generate random numbers in [ -xavier, xavier ]
                Wz.put(y, x, -xavier + Math.random() * 2.0 * xavier);
                Wr.put(y, x, -xavier + Math.random() * 2.0 * xavier);
                Wh.put(y, x, -xavier + Math.random() * 2.0 * xavier);
              }
          }

        xavier = Math.sqrt(6 / (dimState + dimState));
        for(y = 0; y < h; y++)                                      //  Xavier-initialize U matrices
          {
            for(x = 0; x < h; x++)
              {
                                                                    //  Generate random numbers in [ -xavier, xavier ]
                Uz.put(y, x, -xavier + Math.random() * 2.0 * xavier);
                Ur.put(y, x, -xavier + Math.random() * 2.0 * xavier);
                Uh.put(y, x, -xavier + Math.random() * 2.0 * xavier);
              }
          }

        for(x = 0; x < h; x++)                                      //  Initialize b vectors
          {
            bz.put(x, 0.0);
            br.put(x, 0.0);
            bh.put(x, 0.0);
          }

        for(y = 0; y < h; y++)                                      //  Blank out the cache
          {
            for(x = 0; x < cache; x++)
              H.put(y, x, 0.0);
          }

        layerName = nameStr;
      }

    /*  */
    public GRULayer(int dimInput, int dimState, int cacheSize)
      {
        this(dimInput, dimState, cacheSize, "");
      }

    /* Use placeholder arguments */
    public GRULayer()
      {
        this(1, 1, 1, "");
      }

    /* Set the entirety of the Wz matrix using the given array */
    public void setWz(double[] w)
      {
        int x, y;

        for(y = 0; y < h; y++)
          {
            for(x = 0; x < d; x++)
              Wz.put(y, x, w[y * d + x]);
          }

        return;
      }

    /* Set the entirety of the Wr matrix using the given array */
    public void setWr(double[] w)
      {
        int x, y;

        for(y = 0; y < h; y++)
          {
            for(x = 0; x < d; x++)
              Wr.put(y, x, w[y * d + x]);
          }

        return;
      }

    /* Set the entirety of the Wh matrix using the given array */
    public void setWh(double[] w)
      {
        int x, y;

        for(y = 0; y < h; y++)
          {
            for(x = 0; x < d; x++)
              Wh.put(y, x, w[y * d + x]);
          }

        return;
      }

    /* Set column[i], row[j] of the Wz matrix */
    public void setWz_ij(double w, int i, int j)
      {
        if(i < d && j < h)
          Wz.put(j, i, w);
        return;
      }

    /* Set column[i], row[j] of the Wr matrix */
    public void setWr_ij(double w, int i, int j)
      {
        if(i < d && j < h)
          Wr.put(j, i, w);
        return;
      }

    /* Set column[i], row[j] of the Wh matrix */
    public void setWh_ij(double w, int i, int j)
      {
        if(i < d && j < h)
          Wh.put(j, i, w);
        return;
      }

    /* Set the entirety of the Uz matrix using the given array */
    public void setUz(double[] w)
      {
        int x, y;

        for(y = 0; y < h; y++)
          {
            for(x = 0; x < h; x++)
              Uz.put(y, x, w[y * h + x]);
          }

        return;
      }

    /* Set the entirety of the Ur matrix using the given array */
    public void setUr(double[] w)
      {
        int x, y;

        for(y = 0; y < h; y++)
          {
            for(x = 0; x < h; x++)
              Ur.put(y, x, w[y * h + x]);
          }

        return;
      }

    /* Set the entirety of the Uh matrix using the given array */
    public void setUh(double[] w)
      {
        int x, y;

        for(y = 0; y < h; y++)
          {
            for(x = 0; x < h; x++)
              Uh.put(y, x, w[y * h + x]);
          }

        return;
      }

    /* Set column[i], row[j] of the Uz matrix */
    public void setUz_ij(double w, int i, int j)
      {
        if(i < h && j < h)
          Uz.put(j, i, w);
        return;
      }

    /* Set column[i], row[j] of the Ur matrix */
    public void setUr_ij(double w, int i, int j)
      {
        if(i < h && j < h)
          Ur.put(j, i, w);
        return;
      }

    /* Set column[i], row[j] of the Uh matrix */
    public void setUh_ij(double w, int i, int j)
      {
        if(i < h && j < h)
          Uh.put(j, i, w);
        return;
      }

    /* Set the entirety of the bz vector using the given array */
    public void setbz(double[] w)
      {
        int i;
        for(i = 0; i < h; i++)
          bz.put(i, w[i]);
        return;
      }

    /* Set the entirety of the br vector using the given array */
    public void setbr(double[] w)
      {
        int i;
        for(i = 0; i < h; i++)
          br.put(i, w[i]);
        return;
      }

    /* Set the entirety of the bh vector using the given array */
    public void setbh(double[] w)
      {
        int i;
        for(i = 0; i < h; i++)
          bh.put(i, w[i]);
        return;
      }

    /* Set element [i] of the bz vector */
    public void setbz_i(double w, int i)
      {
        if(i < h)
          bz.put(i, w);
        return;
      }

    /* Set element [i] of the br vector */
    public void setbr_i(double w, int i)
      {
        if(i < h)
          br.put(i, w);
        return;
      }

    /* Set element [i] of the bh vector */
    public void setbh_i(double w, int i)
      {
        if(i < h)
          bh.put(i, w);
        return;
      }

    public void setName(String nameStr)
      {
        layerName = nameStr;
        return;
      }

    public void print()
      {
        int i, j;

        System.out.printf("Input dimensionality d = %d\n", d);
        System.out.printf("State dimensionality h = %d\n", h);
        System.out.printf("State cache size       = %d\n", cache);

        System.out.printf("Wz (%d x %d)\n", h, d);
        for(i = 0; i < h; i++)
          {
            System.out.print("[");
            for(j = 0; j < d; j++)
              System.out.printf(" %.5f", Wz.get(i, j));
            System.out.print(" ]\n");
          }
        System.out.printf("Wr (%d x %d)\n", h, d);
        for(i = 0; i < h; i++)
          {
            System.out.print("[");
            for(j = 0; j < d; j++)
              System.out.printf(" %.5f", Wr.get(i, j));
            System.out.print(" ]\n");
          }
        System.out.printf("Wh (%d x %d)\n", h, d);
        for(i = 0; i < h; i++)
          {
            System.out.print("[");
            for(j = 0; j < d; j++)
              System.out.printf(" %.5f", Wh.get(i, j));
            System.out.print(" ]\n");
          }

        System.out.printf("Uz (%d x %d)\n", h, h);
        for(i = 0; i < h; i++)
          {
            System.out.print("[");
            for(j = 0; j < h; j++)
              System.out.printf(" %.5f", Uz.get(i, j));
            System.out.print(" ]\n");
          }
        System.out.printf("Ur (%d x %d)\n", h, h);
        for(i = 0; i < h; i++)
          {
            System.out.print("[");
            for(j = 0; j < h; j++)
              System.out.printf(" %.5f", Ur.get(i, j));
            System.out.print(" ]\n");
          }
        System.out.printf("Uh (%d x %d)\n", h, h);
        for(i = 0; i < h; i++)
          {
            System.out.print("[");
            for(j = 0; j < h; j++)
              System.out.printf(" %.5f", Uh.get(i, j));
            System.out.print(" ]\n");
          }

        System.out.printf("bz (%d x 1)\n", h);
        for(i = 0; i < h; i++)
          System.out.printf("[ %.5f ]\n", bz.get(i));
        System.out.printf("br (%d x 1)\n", h);
        for(i = 0; i < h; i++)
          System.out.printf("[ %.5f ]\n", br.get(i));
        System.out.printf("bh (%d x 1)\n", h);
        for(i = 0; i < h; i++)
          System.out.printf("[ %.5f ]\n", bh.get(i));

        return;
      }

    public int inputDimensionality()
      {
        return d;
      }

    public int stateDimensionality()
      {
        return h;
      }

    public int cacheLen()
      {
        return cache;
      }

    public int timestep()
      {
        return t;
      }

    public String name()
      {
        return layerName;
      }

    /* Return the index-th column vector of the H matrix.
       index = 0 would return the earliest cached state.
       index = cache - 1 would retirn the latest cached state. */
    public double[] state(int index)
      {
        double hVec[];
        int i;

        if(index < cache && index >= 0)
          {
            hVec = new double[h];
            for(i = 0; i < h; i++)
              hVec[i] = H.get(i, index);
            return hVec;
          }
        return null;
      }

    /* Return the LAST column vector of the H matrix.
       (The latest cached state.O */
    public double[] state()
      {
        double hVec[];
        int i;

        hVec = new double[h];
        for(i = 0; i < h; i++)
           hVec[i] = H.get(i, cache - 1);
        return hVec;
      }

    public int outputLen()
      {
        return h;
      }

    /* Run the given input vector 'x' of length 'd' through the GRULayer.
       Output is stored internally in H.
       Write to the 't'-th column and increment t.
       If 't' exceeds 'cache', shift everything down. */
    public int run(double[] x)
      {
        int n, m;

        DoubleMatrix x_vec = new DoubleMatrix(x);                   //  Create a column vector from given array

        DoubleMatrix z_vec = new DoubleMatrix(h);
        DoubleMatrix r_vec = new DoubleMatrix(h);
        DoubleMatrix h_vec = new DoubleMatrix(h);
        DoubleMatrix hprime = new DoubleMatrix(h);                  //  Intermediate Hadamard product r * ht_1
        DoubleMatrix ht_1 = new DoubleMatrix(h);                    //  Time t - 1

        int time_1;                                                 //  Where we READ FROM
        int time;                                                   //  Where we WRITE TO
                                                                    //  layer->t increases indefinitely
        if(t == 0)                                                  //  Timestep layer->t = 0 uses the zero-vectors for t - 1
          {
            time_1 = 0;
            time = 0;
            for(n = 0; n < h; n++)                                  //  Write zeroes to h(t-1)
              ht_1.put(n, 0.0);
          }
        else                                                        //  Timestep t > 0 uses the previous state
          {                                                         //  Consider that we may have shifted states
            if(t >= cache)                                          //  out of the matrix
              {
                time_1 = cache - 1;                                 //  Read from the rightmost column
                                                                    //  (then shift everything left)
                time = cache - 1;                                   //  Write to the rightmost column
              }
            else                                                    //  We've not yet maxed out cache
              {
                time_1 = t - 1;                                     //  Read from the previous column
                time = t;                                           //  Write to the targeted column
              }
            for(n = 0; n < h; n++)
              ht_1.put(n, H.get(n, time_1));
          }

        for(n = 0; n < h; n++)                                      //  Write biases to vectors z and r
          {
            z_vec.put(n, bz.get(n));
            r_vec.put(n, br.get(n));
            h_vec.put(n, 0.0);                                      //  Blank out h
          }

        z_vec.add(Uz.mmul(ht_1));                                   //  Add Uz dot ht_1 to z
        r_vec.add(Ur.mmul(ht_1));                                   //  Add Ur dot ht_1 to r

        z_vec.add(Wz.mmul(x_vec));                                  //  Add Wz dot x to z
        r_vec.add(Wr.mmul(x_vec));                                  //  Add Wr dot x to r

        ActivationFunction.sigmoid(z_vec);                          //  Apply sigmoid function to z and r vectors
        ActivationFunction.sigmoid(r_vec);

        for(n = 0; n < h; n++)                                      //  h' = r * ht_1
          hprime.put(n, r_vec.get(n) * ht_1.get(n));

        h_vec = Uh.mmul(hprime);                                    //  Set h = Uh.h' = Uh.(r * ht_1)
        h_vec.add(Wh.mmul(x_vec));

        for(n = 0; n < h; n++)                                      //  Add bias to h vector
          h_vec.put(n, h_vec.get(n) + bh.get(n));

        //  Now h = Wh.x + Uh.(r * ht_1) + bh

        //  We have allocated h-by-cache space for 'H', but the structure and routine should not crash if
        //  we write more than 'cache' states. Shift everything down one column and write to the end.
        if(t >= cache)
          {
            for(m = 1; m < cache; m++)                              //  Shift down
              {
                for(n = 0; n < h; n++)
                  H.put(n, m - 1, H.get(n, m));
              }
          }

        for(n = 0; n < h; n++)                                      //  h = z*ht_1 + (1-z)*tanh(h)
          H.put(n, time, z_vec.get(n) * ht_1.get(n) + (1.0 - z_vec.get(n) * ActivationFunction.tanh(h_vec.get(n))));

        x_vec = null;                                               //  Force release of allocated memory

        z_vec = null;
        r_vec = null;
        hprime = null;
        h_vec = null;

        ht_1 = null;

        System.gc();                                                //  Call the garbage collector

        t++;                                                        //  Increment time step

        return h;                                                   //  Return the size of the state
      }

    public void reset()
      {
        int x, y;

        t = 0;                                                      //  Reset time step

        for(y = 0; y < h; y++)                                      //  Blank out the cache
          {
            for(x = 0; x < cache; x++)
              H.put(y, x, 0.0);
          }

        return;
      }

    public boolean read(DataInputStream fp)
      {
        int ctr;

        ByteBuffer byteBuffer;
        int allocation;
        byte byteArr[];

        allocation = 12;                                            //  Allocate space for 3 ints
        byteArr = new byte[allocation];

        try
          {
            fp.read(byteArr);
          }
        catch(IOException ioErr)
          {
            System.out.println("ERROR: Unable to read GRU Layer header from file.");
            return false;
          }

        byteBuffer = ByteBuffer.allocate(allocation);
        byteBuffer = ByteBuffer.wrap(byteArr);
        byteBuffer.order(ByteOrder.LITTLE_ENDIAN);                  //  Read little-endian

        d = byteBuffer.getInt();                                    //  (int) Read dimensionality of layer input from file
        h = byteBuffer.getInt();                                    //  (int) Read dimensionality of layer state from file
        cache = byteBuffer.getInt();                                //  (int) Read length of layer cache from file

        Wz = new DoubleMatrix(h, d);                                //  (Re)Allocate this layer's Wz matrix
        Wr = new DoubleMatrix(h, d);                                //  (Re)Allocate this layer's Wr matrix
        Wh = new DoubleMatrix(h, d);                                //  (Re)Allocate this layer's Wh matrix

        Uz = new DoubleMatrix(h, h);                                //  (Re)Allocate this layer's Uz matrix
        Ur = new DoubleMatrix(h, h);                                //  (Re)Allocate this layer's Ur matrix
        Uh = new DoubleMatrix(h, h);                                //  (Re)Allocate this layer's Uh matrix

        bz = new DoubleMatrix(h);                                   //  (Re)Allocate this layer's bz vector
        br = new DoubleMatrix(h);                                   //  (Re)Allocate this layer's br vector
        bh = new DoubleMatrix(h);                                   //  (Re)Allocate this layer's bh vector

        H = new DoubleMatrix(h, cache);                             //  (Re)Allocate this newest layer's cache matrix

        allocation = (h * d + h * h + h) * 24 +                     //  Allocate space for all weight matrices and the layer name
                     NeuralNet.LAYER_NAME_LEN;
        byteArr = new byte[allocation];

        try
          {
            fp.read(byteArr);
          }
        catch(IOException ioErr)
          {
            System.out.println("ERROR: Unable to read GRU Layer weights from file.");
            return false;
          }

        byteBuffer = ByteBuffer.allocate(allocation);
        byteBuffer = ByteBuffer.wrap(byteArr);
        byteBuffer.order(ByteOrder.LITTLE_ENDIAN);                  //  Read little-endian

        for(ctr = 0; ctr < d * h; ctr++)                            //  Read Wz (h * d doubles)
          Wz.put((ctr - (ctr % d)) / d, ctr % d, byteBuffer.getDouble());
        for(ctr = 0; ctr < d * h; ctr++)                            //  Read Wr (h * d doubles)
          Wr.put((ctr - (ctr % d)) / d, ctr % d, byteBuffer.getDouble());
        for(ctr = 0; ctr < d * h; ctr++)                            //  Read Wh (h * d doubles)
          Wh.put((ctr - (ctr % d)) / d, ctr % d, byteBuffer.getDouble());

        for(ctr = 0; ctr < h * h; ctr++)                            //  Read Uz (h * h doubles)
          Uz.put((ctr - (ctr % h)) / h, ctr % h, byteBuffer.getDouble());
        for(ctr = 0; ctr < h * h; ctr++)                            //  Read Ur (h * h doubles)
          Ur.put((ctr - (ctr % h)) / h, ctr % h, byteBuffer.getDouble());
        for(ctr = 0; ctr < h * h; ctr++)                            //  Read Uh (h * h doubles)
          Uh.put((ctr - (ctr % h)) / h, ctr % h, byteBuffer.getDouble());

        for(ctr = 0; ctr < h; ctr++)                                //  Read bz (h doubles)
          bz.put(ctr, byteBuffer.getDouble());
        for(ctr = 0; ctr < h; ctr++)                                //  Read br (h doubles)
          br.put(ctr, byteBuffer.getDouble());
        for(ctr = 0; ctr < h; ctr++)                                //  Read bh (h doubles)
          bh.put(ctr, byteBuffer.getDouble());

        for(ctr = 0; ctr < h * cache; ctr++)                        //  Blank out H matrix
          H.put((ctr - (ctr % cache)) / cache, ctr % cache, 0.0);

        byteArr = new byte[NeuralNet.LAYER_NAME_LEN];               //  Allocate
        for(ctr = 0; ctr < NeuralNet.LAYER_NAME_LEN; ctr++)         //  Read into array
          byteArr[ctr] = byteBuffer.get();
        layerName = new String(byteArr, StandardCharsets.UTF_8);

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
        allocation = 12 + NeuralNet.LAYER_NAME_LEN +                //  3 ints and the layer name,
                     3 * d * h * 8 +                                //  3*d*h doubles at 8 bytes each,
                     3 * h * h * 8 +                                //  3*h*h doubles at 8 bytes each,
                     3 * h * 8;                                     //  3*h doubles at 8 bytes each.
        byteBuffer = ByteBuffer.allocate(allocation);
        byteBuffer.order(ByteOrder.LITTLE_ENDIAN);                  //  Write little-endian

        byteBuffer.putInt(d);                                       //  (int) Save GRULayer's input dimensionality to file
        byteBuffer.putInt(h);                                       //  (int) Save GRULayer's state dimensionality to file
        byteBuffer.putInt(cache);                                   //  (int) Save length of GRULayer's cache to file

        for(ctr = 0; ctr < d * h; ctr++)                            //  (double) Save the ROW-MAJOR Wz matrix to file
          byteBuffer.putDouble(Wz.get((ctr - (ctr % d)) / d, ctr % d));
        for(ctr = 0; ctr < d * h; ctr++)                            //  (double) Save the ROW-MAJOR Wr matrix to file
          byteBuffer.putDouble(Wr.get((ctr - (ctr % d)) / d, ctr % d));
        for(ctr = 0; ctr < d * h; ctr++)                            //  (double) Save the ROW-MAJOR Wh matrix to file
          byteBuffer.putDouble(Wh.get((ctr - (ctr % d)) / d, ctr % d));

        for(ctr = 0; ctr < h * h; ctr++)                            //  (double) Save the ROW-MAJOR Uz matrix to file
          byteBuffer.putDouble(Uz.get((ctr - (ctr % h)) / h, ctr % h));
        for(ctr = 0; ctr < h * h; ctr++)                            //  (double) Save the ROW-MAJOR Ur matrix to file
          byteBuffer.putDouble(Ur.get((ctr - (ctr % h)) / h, ctr % h));
        for(ctr = 0; ctr < h * h; ctr++)                            //  (double) Save the ROW-MAJOR Uh matrix to file
          byteBuffer.putDouble(Uh.get((ctr - (ctr % h)) / h, ctr % h));

        for(ctr = 0; ctr < h; ctr++)                                //  (double) Save the bz vector to file
          byteBuffer.putDouble(bz.get(ctr));
        for(ctr = 0; ctr < h; ctr++)                                //  (double) Save the br vector to file
          byteBuffer.putDouble(br.get(ctr));
        for(ctr = 0; ctr < h; ctr++)                                //  (double) Save the bh vector to file
          byteBuffer.putDouble(bh.get(ctr));

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
            System.out.println("ERROR: Unable to write GRU Layer to file.");
            return false;
          }

        byteArr = null;                                             //  Release the array
        System.gc();                                                //  Call the garbage collector

        return true;
      }
  }
