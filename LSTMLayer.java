/**************************************************************************************************
 Neural Network library, by Eric C. Joyce

 Model an LSTM Layer as matrices and vectors:
  d = the length of a single input instance
      (that is, we may have an indefinitely long sequence of word-vectors, but each is an input instance of length 'd')
  h = the length of internal state vectors
  cache = the number of previous states to track and store

  input d-vec{x}       weights Wi              weights Wo              weights Wf              weights Wc
 [ x1 x2 x3 x4 ]        (h by d)                (h by d)                (h by d)                (h by d)
                 [ wi11 wi12 wi13 wi14 ] [ wo11 wo12 wo13 wo14 ] [ wf11 wf12 wf13 wf14 ] [ wc11 wc12 wc13 wc14 ]
                 [ wi21 wi22 wi23 wi24 ] [ wo21 wo22 wo23 wo24 ] [ wf21 wf22 wf23 wf24 ] [ wc21 wc22 wc23 wc24 ]
                 [ wi31 wi32 wi33 wi34 ] [ wo31 wo32 wo33 wo34 ] [ wf31 wf32 wf33 wf34 ] [ wc31 wc32 wc33 wc34 ]

                       weights Ui              weights Uo              weights Uf              weights Uc
                        (h by h)                (h by h)                (h by h)                (h by h)
                 [ ui11 ui12 ui13 ]      [ uo11 uo12 uo13 ]      [ uf11 uf12 uf13 ]      [ uc11 uc12 uc13 ]
                 [ ui21 ui22 ui23 ]      [ uo21 uo22 uo23 ]      [ uf21 uf22 uf23 ]      [ uc21 uc22 uc23 ]
                 [ ui31 ui32 ui33 ]      [ uo31 uo32 uo33 ]      [ uf31 uf32 uf33 ]      [ uc31 uc32 uc33 ]

                     bias h-vec{bi}          bias h-vec{bo}          bias h-vec{bf}          bias h-vec{bc}
                 [ bi1 ]                 [ bo1 ]                 [ bf1 ]                 [ bc1 ]
                 [ bi2 ]                 [ bo2 ]                 [ bf2 ]                 [ bc2 ]
                 [ bi3 ]                 [ bo3 ]                 [ bf3 ]                 [ bc3 ]

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
import java.nio.charset.StandardCharsets;
import org.jblas.*;
import static org.jblas.DoubleMatrix.*;

public class LSTMLayer
  {
    private int d;                                                  //  Dimensionality of input vector
    private int h;                                                  //  Dimensionality of hidden state vector
    private int cache;                                              //  The number of states to keep in memory:
                                                                    //  when 't' exceeds this, shift out.
    private int t;                                                  //  The time step
                                                                    //  W matrices are (h by d)
    private DoubleMatrix Wi;                                        //  Input gate weights
    private DoubleMatrix Wo;                                        //  Output gate weights
    private DoubleMatrix Wf;                                        //  Forget gate weights
    private DoubleMatrix Wc;                                        //  Memory cell weights
                                                                    //  U matrices are (h by h)
    private DoubleMatrix Ui;                                        //  Recurrent connection input gate weights
    private DoubleMatrix Uo;                                        //  Recurrent connection output gate weights
    private DoubleMatrix Uf;                                        //  Recurrent connection forget gate weights
    private DoubleMatrix Uc;                                        //  Recurrent connection memory cell weights
                                                                    //  Bias vectors are length h
    private DoubleMatrix bi;                                        //  Input gate bias
    private DoubleMatrix bo;                                        //  Output gate bias
    private DoubleMatrix bf;                                        //  Forget gate bias
    private DoubleMatrix bc;                                        //  Memory cell bias

    private DoubleMatrix c;                                         //  Cell state vector, length h
    private DoubleMatrix H;                                         //  Hidden state cache matrix (h by cache)
    private String layerName;

    /*  */
    public LSTMLayer(int dimInput, int dimState, int cacheSize, String nameStr)
      {
        int x, y;
        double xavier;

        d = dimInput;                                               //  Save dimensionality of input
        h = dimState;                                               //  Save dimensionality of states
        cache = cacheSize;                                          //  Save the cache size
        t = 0;                                                      //  Initial time step = 0

        Wi = new DoubleMatrix(dimState, dimInput);                  //  Allocate this layer's W matrices
        Wo = new DoubleMatrix(dimState, dimInput);
        Wf = new DoubleMatrix(dimState, dimInput);
        Wc = new DoubleMatrix(dimState, dimInput);

        Ui = new DoubleMatrix(dimState, dimState);                  //  Allocate this layer's U matrices
        Uo = new DoubleMatrix(dimState, dimState);
        Uf = new DoubleMatrix(dimState, dimState);
        Uc = new DoubleMatrix(dimState, dimState);

        bi = new DoubleMatrix(dimState);                            //  Allocate this newest layer's b vectors
        bo = new DoubleMatrix(dimState);
        bf = new DoubleMatrix(dimState);
        bc = new DoubleMatrix(dimState);

        c = new DoubleMatrix(dimState);                             //  Allocate this newest layer's c vector

        H = new DoubleMatrix(dimState, cacheSize);                  //  Allocate this newest layer's cache matrix

        xavier = Math.sqrt(6 / (dimInput + dimState));
        for(y = 0; y < h; y++)                                      //  Xavier-initialize W matrices
          {
            for(x = 0; x < d; x++)
              {
                                                                    //  Generate random numbers in [ -xavier, xavier ]
                Wi.put(y, x, -xavier + Math.random() * 2.0 * xavier);
                Wo.put(y, x, -xavier + Math.random() * 2.0 * xavier);
                Wf.put(y, x, -xavier + Math.random() * 2.0 * xavier);
                Wc.put(y, x, -xavier + Math.random() * 2.0 * xavier);
              }
          }

        xavier = Math.sqrt(6 / (dimState + dimState));
        for(y = 0; y < h; y++)                                      //  Xavier-initialize U matrices
          {
            for(x = 0; x < h; x++)
              {
                                                                    //  Generate random numbers in [ -xavier, xavier ]
                Ui.put(y, x, -xavier + Math.random() * 2.0 * xavier);
                Uo.put(y, x, -xavier + Math.random() * 2.0 * xavier);
                Uf.put(y, x, -xavier + Math.random() * 2.0 * xavier);
                Uc.put(y, x, -xavier + Math.random() * 2.0 * xavier);
              }
          }

        for(x = 0; x < h; x++)                                      //  Initialize b vectors
          {
            bi.put(x, 0.0);
            bo.put(x, 0.0);
            bf.put(x, 1.0);                                         //  Forget gate set to ones
            bc.put(x, 0.0);
          }

        for(y = 0; y < h; y++)                                      //  Blank out the cache
          {
            for(x = 0; x < cache; x++)
              H.put(y, x, 0.0);
          }

        for(x = 0; x < h; x++)                                      //  Blank out the carry
          c.put(x, 0.0);

        layerName = nameStr;
      }

    /*  */
    public LSTMLayer(int dimInput, int dimState, int cacheSize)
      {
        this(dimInput, dimState, cacheSize, "");
      }

    /* Set the entirety of the Wi matrix using the given array */
    public void setWi(double[] w)
      {
        int x, y;

        for(y = 0; y < h; y++)
          {
            for(x = 0; x < d; x++)
              Wi.put(y, x, w[y * d + x]);
          }

        return;
      }

    /* Set the entirety of the Wo matrix using the given array */
    public void setWo(double[] w)
      {
        int x, y;

        for(y = 0; y < h; y++)
          {
            for(x = 0; x < d; x++)
              Wo.put(y, x, w[y * d + x]);
          }

        return;
      }

    /* Set the entirety of the Wf matrix using the given array */
    public void setWf(double[] w)
      {
        int x, y;

        for(y = 0; y < h; y++)
          {
            for(x = 0; x < d; x++)
              Wf.put(y, x, w[y * d + x]);
          }

        return;
      }

    /* Set the entirety of the Wc matrix using the given array */
    public void setWc(double[] w)
      {
        int x, y;

        for(y = 0; y < h; y++)
          {
            for(x = 0; x < d; x++)
              Wc.put(y, x, w[y * d + x]);
          }

        return;
      }

    /* Set column[i], row[j] of the Wi matrix */
    public void setWi_ij(double w, int i, int j)
      {
        if(i < d && j < h)
          Wi.put(j, i, w);
        return;
      }

    /* Set column[i], row[j] of the Wo matrix */
    public void setWo_ij(double w, int i, int j)
      {
        if(i < d && j < h)
          Wo.put(j, i, w);
        return;
      }

    /* Set column[i], row[j] of the Wf matrix */
    public void setWf_ij(double w, int i, int j)
      {
        if(i < d && j < h)
          Wf.put(j, i, w);
        return;
      }

    /* Set column[i], row[j] of the Wc matrix */
    public void setWc_ij(double w, int i, int j)
      {
        if(i < d && j < h)
          Wc.put(j, i, w);
        return;
      }

    /* Set the entirety of the Ui matrix using the given array */
    public void setUi(double[] w)
      {
        int x, y;

        for(y = 0; y < h; y++)
          {
            for(x = 0; x < h; x++)
              Ui.put(y, x, w[y * h + x]);
          }

        return;
      }

    /* Set the entirety of the Uo matrix using the given array */
    public void setUo(double[] w)
      {
        int x, y;

        for(y = 0; y < h; y++)
          {
            for(x = 0; x < h; x++)
              Uo.put(y, x, w[y * h + x]);
          }

        return;
      }

    /* Set the entirety of the Uf matrix using the given array */
    public void setUf(double[] w)
      {
        int x, y;

        for(y = 0; y < h; y++)
          {
            for(x = 0; x < h; x++)
              Uf.put(y, x, w[y * h + x]);
          }

        return;
      }

    /* Set the entirety of the Uc matrix using the given array */
    public void setUc(double[] w)
      {
        int x, y;

        for(y = 0; y < h; y++)
          {
            for(x = 0; x < h; x++)
              Uc.put(y, x, w[y * h + x]);
          }

        return;
      }

    /* Set column[i], row[j] of the Ui matrix */
    public void setUi_ij(double w, int i, int j)
      {
        if(i < h && j < h)
          Ui.put(j, i, w);
        return;
      }

    /* Set column[i], row[j] of the Uo matrix */
    public void setUo_ij(double w, int i, int j)
      {
        if(i < h && j < h)
          Uo.put(j, i, w);
        return;
      }

    /* Set column[i], row[j] of the Uf matrix */
    public void setUf_ij(double w, int i, int j)
      {
        if(i < h && j < h)
          Uf.put(j, i, w);
        return;
      }

    /* Set column[i], row[j] of the Uc matrix */
    public void setUc_ij(double w, int i, int j)
      {
        if(i < h && j < h)
          Uc.put(j, i, w);
        return;
      }

    /* Set the entirety of the bi vector using the given array */
    public void setbi(double[] w)
      {
        int i;
        for(i = 0; i < h; i++)
          bi.put(i, w[i]);
        return;
      }

    /* Set the entirety of the bo vector using the given array */
    public void setbo(double[] w)
      {
        int i;
        for(i = 0; i < h; i++)
          bo.put(i, w[i]);
        return;
      }

    /* Set the entirety of the bf vector using the given array */
    public void setbf(double[] w)
      {
        int i;
        for(i = 0; i < h; i++)
          bf.put(i, w[i]);
        return;
      }

    /* Set the entirety of the bc vector using the given array */
    public void setbc(double[] w)
      {
        int i;
        for(i = 0; i < h; i++)
          bc.put(i, w[i]);
        return;
      }

    /* Set element [i] of the bi vector */
    public void setbi_i(double w, int i)
      {
        if(i < h)
          bi.put(i, w);
        return;
      }

    /* Set element [i] of the bo vector */
    public void setbo_i(double w, int i)
      {
        if(i < h)
          bo.put(i, w);
        return;
      }

    /* Set element [i] of the bf vector */
    public void setbf_i(double w, int i)
      {
        if(i < h)
          bf.put(i, w);
        return;
      }

    /* Set element [i] of the bc vector */
    public void setbc_i(double w, int i)
      {
        if(i < h)
          bc.put(i, w);
        return;
      }

    public void setName(String nameStr)
      {
        layerName = nameStr;
        return;
      }

    /*  */
    public void print()
      {
        int i, j;

        System.out.printf("Input dimensionality d = %d\n", d);
        System.out.printf("State dimensionality h = %d\n", h);
        System.out.printf("State cache size       = %d\n", cache);

        System.out.printf("Wi (%d x %d)\n", h, d);
        for(i = 0; i < d; i++)
          {
            System.out.print("[");
            for(j = 0; j < h; j++)
              System.out.printf(" %.5f", Wi.get(i, j));
            System.out.print(" ]\n");
          }
        System.out.printf("Wf (%d x %d)\n", h, d);
        for(i = 0; i < d; i++)
          {
            System.out.print("[");
            for(j = 0; j < h; j++)
              System.out.printf(" %.5f", Wf.get(i, j));
            System.out.print(" ]\n");
          }
        System.out.printf("Wc (%d x %d)\n", h, d);
        for(i = 0; i < d; i++)
          {
            System.out.print("[");
            for(j = 0; j < h; j++)
              System.out.printf(" %.5f", Wc.get(i, j));
            System.out.print(" ]\n");
          }
        System.out.printf("Wo (%d x %d)\n", h, d);
        for(i = 0; i < d; i++)
          {
            System.out.print("[");
            for(j = 0; j < h; j++)
              System.out.printf(" %.5f", Wo.get(i, j));
            System.out.print(" ]\n");
          }

        System.out.printf("Ui (%d x %d)\n", h, h);
        for(i = 0; i < h; i++)
          {
            System.out.print("[");
            for(j = 0; j < h; j++)
              System.out.printf(" %.5f", Ui.get(i, j));
            System.out.print(" ]\n");
          }
        System.out.printf("Uf (%d x %d)\n", h, h);
        for(i = 0; i < h; i++)
          {
            System.out.print("[");
            for(j = 0; j < h; j++)
              System.out.printf(" %.5f", Uf.get(i, j));
            System.out.print(" ]\n");
          }
        System.out.printf("Uc (%d x %d)\n", h, h);
        for(i = 0; i < h; i++)
          {
            System.out.print("[");
            for(j = 0; j < h; j++)
              System.out.printf(" %.5f", Uc.get(i, j));
            System.out.print(" ]\n");
          }
        System.out.printf("Uo (%d x %d)\n", h, h);
        for(i = 0; i < h; i++)
          {
            System.out.print("[");
            for(j = 0; j < h; j++)
              System.out.printf(" %.5f", Uo.get(i, j));
            System.out.print(" ]\n");
          }

        System.out.printf("bi (%d x 1)\n", h);
        for(i = 0; i < h; i++)
          System.out.printf("[ %.5f ]\n", bi.get(i));
        System.out.printf("bf (%d x 1)\n", h);
        for(i = 0; i < h; i++)
          System.out.printf("[ %.5f ]\n", bf.get(i));
        System.out.printf("bc (%d x 1)\n", h);
        for(i = 0; i < h; i++)
          System.out.printf("[ %.5f ]\n", bc.get(i));
        System.out.printf("bo (%d x 1)\n", h);
        for(i = 0; i < h; i++)
          System.out.printf("[ %.5f ]\n", bo.get(i));

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

    /*  */
    public int outputLen()
      {
        return h;
      }

    /* Run the given input vector 'x' of length 'd' through the LSTMLayer.
       Output is stored internally in H.
       Write to the 't'-th column and increment t.
       If 't' exceeds 'cache', shift everything down. */
    public int run(double[] x)
      {
        int n, m;

        DoubleMatrix x_vec = new DoubleMatrix(x);                   //  Create a column vector from given array

        DoubleMatrix i_vec = new DoubleMatrix(h);
        DoubleMatrix f_vec = new DoubleMatrix(h);
        DoubleMatrix c_vec = new DoubleMatrix(h);                   //  Time t
        DoubleMatrix o_vec = new DoubleMatrix(h);

        DoubleMatrix ct_1 = new DoubleMatrix(h);                    //  Time t - 1
        DoubleMatrix ht_1 = new DoubleMatrix(h);                    //  Time t - 1

        int time_1;                                                 //  Where we READ FROM
        int time;                                                   //  Where we WRITE TO
                                                                    //  t increases indefinitely
        if(t == 0)                                                  //  Timestep t = 0 uses the zero-vectors for t - 1
          {
            time_1 = 0;
            time = 0;
            for(n = 0; n < h; n++)                                  //  Write zeroes to h(t-1) and c(t-1)
              {
                ht_1.put(n, 0.0);
                ct_1.put(n, 0.0);
              }
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
              {
                ht_1.put(n, H.get(n, time_1));
                ct_1.put(n, c.get(n));
              }
          }

        for(n = 0; n < h; n++)                                      //  Write biases to vectors
          {
            i_vec.put(n, bi.get(n));
            f_vec.put(n, bf.get(n));
            c_vec.put(n, bc.get(n));
            o_vec.put(n, bo.get(n));
          }

        i_vec.add(Ui.mmul(ht_1));                                   //  Add Ui dot ht_1 to i
        f_vec.add(Uf.mmul(ht_1));                                   //  Add Uf dot ht_1 to f
        c_vec.add(Uc.mmul(ht_1));                                   //  Add Uc dot ht_1 to c
        o_vec.add(Uo.mmul(ht_1));                                   //  Add Uo dot ht_1 to o

        i_vec.add(Wi.mmul(x_vec));                                  //  Add Wi dot x to i
        f_vec.add(Wf.mmul(x_vec));                                  //  Add Wf dot x to f
        c_vec.add(Wc.mmul(x_vec));                                  //  Add Wc dot x to c
        o_vec.add(Wo.mmul(x_vec));                                  //  Add Wo dot x to o

        //  We have allocated h-by-cache space for 'H', but the structure and routine should not crash if
        //  we write more than 'cache' states. Shift everything down one column and write to the end.
        if(t >= cache)
          {
            for(m = 1; m < cache; m++)                              //  Shift down
              {
                for(n = 0; n < h; n++)
                  H.put(m - 1, n, H.get(m, n));
              }
          }

        for(n = 0; n < h; n++)
          {
            i_vec.put(n, ActivationFunction.sigmoid(i_vec.get(n))); //  i = sig(Wi*x + Ui*ht_1 + bi)
            f_vec.put(n, ActivationFunction.sigmoid(f_vec.get(n))); //  f = sig(Wf*x + Uf*ht_1 + bf)
                                                                    //  c = f*ct_1 + i*tanh(Wc*x + Uc*ht_1 + bc)
            c_vec.put(n, f_vec.get(n) * ct_1.get(n) + i_vec.get(n) * ActivationFunction.tanh(c_vec.get(n)));
            o_vec.put(n, ActivationFunction.sigmoid(o_vec.get(n))); //  o = sig(Wo*x + Uo*ht_1 + bo)
                                                                    //  h = o*tanh(c)
            H.put(n, time, o_vec.get(n) * ActivationFunction.tanh(c_vec.get(n)));
          }

        x_vec = null;                                               //  Force release of allocated memory

        i_vec = null;
        f_vec = null;
        c_vec = null;
        o_vec = null;

        ct_1 = null;
        ht_1 = null;

        System.gc();                                                //  Call the garbage collector

        t++;                                                        //  Increment time step

        return h;                                                   //  Return the size of the state
      }

    /*  */
    public void reset()
      {
        int x, y;

        t = 0;                                                      //  Reset time step

        for(y = 0; y < h; y++)                                      //  Blank out the cache
          {
            for(x = 0; x < cache; x++)
              H.put(y, x, 0.0);
          }

        for(x = 0; x < h; x++)                                      //  Blank out the carry
          c.put(x, 0.0);

        return;
      }

    public boolean read(DataInputStream fp)
      {
        int ctr;
        byte buffer[];

        try
          {
            d = fp.readInt();                                       //  (int) Read dimensionality of layer input from file
          }
        catch(IOException ioErr)
          {
            System.out.println("ERROR: Unable to read dimensionality of LSTM Layer input.");
            return false;
          }

        try
          {
            h = fp.readInt();                                       //  (int) Read dimensionality of layer state from file
          }
        catch(IOException ioErr)
          {
            System.out.println("ERROR: Unable to read dimensionality of LSTM Layer state.");
            return false;
          }

        try
          {
            cache = fp.readInt();                                    //  (int) Read length of layer cache from file
          }
        catch(IOException ioErr)
          {
            System.out.println("ERROR: Unable to read length of LSTM Layer cache.");
            return false;
          }

        Wi = new DoubleMatrix(h, d);                                //  (Re)Allocate
        Wo = new DoubleMatrix(h, d);
        Wf = new DoubleMatrix(h, d);
        Wc = new DoubleMatrix(h, d);

        Ui = new DoubleMatrix(h, h);                                //  (Re)Allocate
        Uo = new DoubleMatrix(h, h);
        Uf = new DoubleMatrix(h, h);
        Uc = new DoubleMatrix(h, h);

        bi = new DoubleMatrix(h);                                   //  (Re)Allocate
        bo = new DoubleMatrix(h);
        bf = new DoubleMatrix(h);
        bc = new DoubleMatrix(h);

        c = new DoubleMatrix(h);
        H = new DoubleMatrix(h, cache);

        for(ctr = 0; ctr < h * d; ctr++)                            //  Read Wi (d * h doubles)
          {
            try
              {
                Wi.put((ctr - (ctr % d)) / d, ctr % d, fp.readDouble());
              }
            catch(IOException ioErr)
              {
                System.out.println("ERROR: Unable to read LSTM Layer Wi weights.");
                return false;
              }
          }
        for(ctr = 0; ctr < h * d; ctr++)                            //  Read Wo (d * h doubles)
          {
            try
              {
                Wo.put((ctr - (ctr % d)) / d, ctr % d, fp.readDouble());
              }
            catch(IOException ioErr)
              {
                System.out.println("ERROR: Unable to read LSTM Layer Wo weights.");
                return false;
              }
          }
        for(ctr = 0; ctr < h * d; ctr++)                            //  Read Wf (d * h doubles)
          {
            try
              {
                Wf.put((ctr - (ctr % d)) / d, ctr % d, fp.readDouble());
              }
            catch(IOException ioErr)
              {
                System.out.println("ERROR: Unable to read LSTM Layer Wf weights.");
                return false;
              }
          }
        for(ctr = 0; ctr < h * d; ctr++)                            //  Read Wc (d * h doubles)
          {
            try
              {
                Wc.put((ctr - (ctr % d)) / d, ctr % d, fp.readDouble());
              }
            catch(IOException ioErr)
              {
                System.out.println("ERROR: Unable to read LSTM Layer Wc weights.");
                return false;
              }
          }

        for(ctr = 0; ctr < h * h; ctr++)                            //  Read Ui (h * h doubles)
          {
            try
              {
                Ui.put((ctr - (ctr % h)) / h, ctr % h, fp.readDouble());
              }
            catch(IOException ioErr)
              {
                System.out.println("ERROR: Unable to read LSTM Layer Ui weights.");
                return false;
              }
          }
        for(ctr = 0; ctr < h * h; ctr++)                            //  Read Uo (h * h doubles)
          {
            try
              {
                Uo.put((ctr - (ctr % h)) / h, ctr % h, fp.readDouble());
              }
            catch(IOException ioErr)
              {
                System.out.println("ERROR: Unable to read LSTM Layer Uo weights.");
                return false;
              }
          }
        for(ctr = 0; ctr < h * h; ctr++)                            //  Read Uf (h * h doubles)
          {
            try
              {
                Uf.put((ctr - (ctr % h)) / h, ctr % h, fp.readDouble());
              }
            catch(IOException ioErr)
              {
                System.out.println("ERROR: Unable to read LSTM Layer Uf weights.");
                return false;
              }
          }
        for(ctr = 0; ctr < h * h; ctr++)                            //  Read Uc (h * h doubles)
          {
            try
              {
                Uc.put((ctr - (ctr % h)) / h, ctr % h, fp.readDouble());
              }
            catch(IOException ioErr)
              {
                System.out.println("ERROR: Unable to read LSTM Layer Uc weights.");
                return false;
              }
          }

        for(ctr = 0; ctr < h; ctr++)                                //  Read bi (h doubles)
          {
            try
              {
                bi.put(ctr, fp.readDouble());
              }
            catch(IOException ioErr)
              {
                System.out.println("ERROR: Unable to read LSTM Layer bi weights.");
                return false;
              }
          }
        for(ctr = 0; ctr < h; ctr++)                                //  Read bo (h doubles)
          {
            try
              {
                bo.put(ctr, fp.readDouble());
              }
            catch(IOException ioErr)
              {
                System.out.println("ERROR: Unable to read LSTM Layer bo weights.");
                return false;
              }
          }
        for(ctr = 0; ctr < h; ctr++)                                //  Read bf (h doubles)
          {
            try
              {
                bf.put(ctr, fp.readDouble());
              }
            catch(IOException ioErr)
              {
                System.out.println("ERROR: Unable to read LSTM Layer bf weights.");
                return false;
              }
          }
        for(ctr = 0; ctr < h; ctr++)                                //  Read bc (h doubles)
          {
            try
              {
                bc.put(ctr, fp.readDouble());
              }
            catch(IOException ioErr)
              {
                System.out.println("ERROR: Unable to read LSTM Layer bc weights.");
                return false;
              }
          }

        for(ctr = 0; ctr < h; ctr++)                                //  Blank out c vector
          c.put(ctr, 0.0);

        for(ctr = 0; ctr < h * cache; ctr++)                        //  Blank out H matrix
          H.put((ctr - (ctr % cache)) / cache, ctr % cache, 0.0);

        buffer = new byte[NeuralNet.LAYER_NAME_LEN];                //  Allocate
        for(ctr = 0; ctr < NeuralNet.LAYER_NAME_LEN; ctr++)         //  Read layerName (NeuralNet.LAYER_NAME_LEN chars)
          {
            try
              {
                buffer[ctr] = fp.readByte();
              }
            catch(IOException ioErr)
              {
                System.out.println("ERROR: Unable to read LSTM Layer name.");
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
            fp.writeInt(d);                                         //  (int) Write dimensionality of layer input to file
          }
        catch(IOException ioErr)
          {
            System.out.println("ERROR: Unable to write dimensionality of LSTM Layer input.");
            return false;
          }

        try
          {
            fp.writeInt(h);                                         //  (int) Write dimensionality of layer state to file
          }
        catch(IOException ioErr)
          {
            System.out.println("ERROR: Unable to write dimensionality of LSTM Layer state.");
            return false;
          }

        try
          {
            fp.writeInt(cache);                                     //  (int) Write length of layer cache to file
          }
        catch(IOException ioErr)
          {
            System.out.println("ERROR: Unable to write length of LSTM Layer cache.");
            return false;
          }

        for(ctr = 0; ctr < h * d; ctr++)                            //  Write Wi (d * h doubles)
          {
            try
              {
                fp.writeDouble(Wi.get((ctr - (ctr % d)) / d, ctr % d));
              }
            catch(IOException ioErr)
              {
                System.out.println("ERROR: Unable to write LSTM Layer Wi weights.");
                return false;
              }
          }
        for(ctr = 0; ctr < h * d; ctr++)                            //  Write Wo (d * h doubles)
          {
            try
              {
                fp.writeDouble(Wo.get((ctr - (ctr % d)) / d, ctr % d));
              }
            catch(IOException ioErr)
              {
                System.out.println("ERROR: Unable to write LSTM Layer Wo weights.");
                return false;
              }
          }
        for(ctr = 0; ctr < h * d; ctr++)                            //  Write Wf (d * h doubles)
          {
            try
              {
                fp.writeDouble(Wf.get((ctr - (ctr % d)) / d, ctr % d));
              }
            catch(IOException ioErr)
              {
                System.out.println("ERROR: Unable to write LSTM Layer Wf weights.");
                return false;
              }
          }
        for(ctr = 0; ctr < h * d; ctr++)                            //  Write Wc (d * h doubles)
          {
            try
              {
                fp.writeDouble(Wc.get((ctr - (ctr % d)) / d, ctr % d));
              }
            catch(IOException ioErr)
              {
                System.out.println("ERROR: Unable to write LSTM Layer Wc weights.");
                return false;
              }
          }

        for(ctr = 0; ctr < h * h; ctr++)                            //  Write Ui (h * h doubles)
          {
            try
              {
                fp.writeDouble(Ui.get((ctr - (ctr % h)) / h, ctr % h));
              }
            catch(IOException ioErr)
              {
                System.out.println("ERROR: Unable to write LSTM Layer Ui weights.");
                return false;
              }
          }
        for(ctr = 0; ctr < h * h; ctr++)                            //  Write Uo (h * h doubles)
          {
            try
              {
                fp.writeDouble(Uo.get((ctr - (ctr % h)) / h, ctr % h));
              }
            catch(IOException ioErr)
              {
                System.out.println("ERROR: Unable to write LSTM Layer Uo weights.");
                return false;
              }
          }
        for(ctr = 0; ctr < h * h; ctr++)                            //  Write Uf (h * h doubles)
          {
            try
              {
                fp.writeDouble(Uf.get((ctr - (ctr % h)) / h, ctr % h));
              }
            catch(IOException ioErr)
              {
                System.out.println("ERROR: Unable to write LSTM Layer Uf weights.");
                return false;
              }
          }
        for(ctr = 0; ctr < h * h; ctr++)                            //  Write Uc (h * h doubles)
          {
            try
              {
                fp.writeDouble(Uc.get((ctr - (ctr % h)) / h, ctr % h));
              }
            catch(IOException ioErr)
              {
                System.out.println("ERROR: Unable to write LSTM Layer Uc weights.");
                return false;
              }
          }

        for(ctr = 0; ctr < h; ctr++)                                //  Write bi (h doubles)
          {
            try
              {
                fp.writeDouble(bi.get(ctr));
              }
            catch(IOException ioErr)
              {
                System.out.println("ERROR: Unable to write LSTM Layer bi weights.");
                return false;
              }
          }
        for(ctr = 0; ctr < h; ctr++)                                //  Write bo (h doubles)
          {
            try
              {
                fp.writeDouble(bo.get(ctr));
              }
            catch(IOException ioErr)
              {
                System.out.println("ERROR: Unable to write LSTM Layer bo weights.");
                return false;
              }
          }
        for(ctr = 0; ctr < h; ctr++)                                //  Write bf (h doubles)
          {
            try
              {
                fp.writeDouble(bf.get(ctr));
              }
            catch(IOException ioErr)
              {
                System.out.println("ERROR: Unable to write LSTM Layer bf weights.");
                return false;
              }
          }
        for(ctr = 0; ctr < h; ctr++)                                //  Write bc (h doubles)
          {
            try
              {
                fp.writeDouble(bc.get(ctr));
              }
            catch(IOException ioErr)
              {
                System.out.println("ERROR: Unable to write LSTM Layer bc weights.");
                return false;
              }
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
                System.out.println("ERROR: Unable to write LSTM Layer name to file.");
                return false;
              }
          }

        buffer = null;                                              //  Release the array
        System.gc();                                                //  Call the garbage collector

        return true;
      }
  }
