/**************************************************************************************************
 Neural Network library, by Eric C. Joyce

 Model a Pooling Layer as 2D input dimensions and an array of 2D pools.
  inputW = width of the input
  inputH = height of the input

 Each pool has a 2D shape, two dimensions for stride, and a function/type:
  stride_h = horizontal stride of the pool
  stride_v = vertical stride of the pool
  f = {MAX_POOL, AVG_POOL, MIN_POOL, MEDIAN_POOL}

    input mat{X}          pool     output for s = (1, 1)     output for s = (2, 2)
 [ x11 x12 x13 x14 ]    [ . . ]   [ y11  y12  y13 ]         [ y11  y12 ]
 [ x21 x22 x23 x24 ]    [ . . ]   [ y21  y22  y23 ]         [ y21  y22 ]
 [ x31 x32 x33 x34 ]              [ y31  y32  y33 ]
 [ x41 x42 x43 x44 ]              [ y41  y42  y43 ]
 [ x51 x52 x53 x54 ]

 Pools needn't be arranged from smallest to largest or in any order.

 Note that this file does NOT seed the randomizer. That should be done by the parent program.
***************************************************************************************************/

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;

public class PoolLayer
  {
    public static final int MAX_POOL    = 0;
    public static final int MIN_POOL    = 1;
    public static final int AVG_POOL    = 2;
    public static final int MEDIAN_POOL = 3;

    private int inputW, inputH;                                     //  Dimensions of the input
    private Pool2D pools[];                                         //  Array of 2D pool objects
    private int n;                                                  //  Length of that array

    private double out[];                                           //  Output buffer
    private int outlen;                                             //  Length of the buffer
    private String layerName;

    public PoolLayer(int w, int h, String nameStr)
      {
        inputW = w;                                                 //  Set this newest layer's input dimensions
        inputH = h;
        n = 0;                                                      //  New layer initially contains zero pools
        outlen = 0;
        layerName = nameStr;
      }

    public PoolLayer(int w, int h)
      {
        this(w, h, "");
      }

    /* Use placeholder arguments */
    public PoolLayer()
      {
        this(1, 1, "");
      }

    /* (Re)Set the width of the i-th pool in this layer. */
    public void setW(int w, int i)
      {
        if(i < n && w > 0)
          {
            pools[i].w = w;

            outlen = outputLen();                                   //  Re-compute layer's output length and reallocate its output buffer
            out = new double[outlen];                               //  Re-allocate this layer's output array
          }
        return;
      }

    /* (Re)Set the height of the i-th pool in this layer. */
    public void setH(int h, int i)
      {
        if(i < n && h > 0)
          {
            pools[i].h = h;

            outlen = outputLen();                                   //  Re-compute layer's output length and reallocate its output buffer
            out = new double[outlen];                               //  Re-allocate this layer's output array
          }
        return;
      }

    /* Set the horizontal stride for the i-th pool in this layer. */
    public void setHorzStride(int stride, int i)
      {
        if(i < n && stride > 0)
          {
            pools[i].stride_h = stride;

            outlen = outputLen();                                   //  Re-compute layer's output length and reallocate its output buffer
            out = new double[outlen];                               //  Re-allocate this layer's output array
          }
        return;
      }

    /* Set the vertical stride for the i-th pool in this layer. */
    public void setVertStride(int stride, int i)
      {
        if(i < n && stride > 0)
          {
            pools[i].stride_v = stride;

            outlen = outputLen();                                   //  Re-compute layer's output length and reallocate its output buffer
            out = new double[outlen];                               //  Re-allocate this layer's output array
          }
        return;
      }

    /* Set the function for the i-th pool in this layer. */
    public void setFunc(int func, int i)
      {
        if(i < n && func >= 0)
          {
            pools[i].f = func;
          }
        return;
      }

    /* Set the name of this Pooling Layer */
    public void setName(String nameStr)
      {
        layerName = nameStr;
        return;
      }

    /* Print the details of this Pool2DLayer */
    public void print()
      {
        int i, x, y;

        System.out.printf("Input Shape = (%d, %d)\n", inputW, inputH);
        for(i = 0; i < n; i++)                                      //  Draw each pool
          {
            System.out.printf("Pool %d\n", i);
            for(y = 0; y < pools[i].h; y++)
              {
                System.out.print("  [");
                for(x = 0; x < pools[i].w; x++)
                  System.out.print(" . ");
                System.out.print("]");
                if(y < pools[i].h - 1)
                  System.out.print("\n");
              }
            System.out.print("  Func:  ");
            switch(pools[i].f)
              {
                case MAX_POOL:     System.out.print("max.  ");  break;
                case MIN_POOL:     System.out.print("min.  ");  break;
                case AVG_POOL:     System.out.print("avg.  ");  break;
                case MEDIAN_POOL:  System.out.print("med.  ");  break;
              }
            System.out.printf("Stride: (%d, %d)\n", pools[i].stride_h, pools[i].stride_v);
          }
        return;
      }

    public int width()
      {
        return inputW;
      }

    public int height()
      {
        return inputH;
      }

    public int numPools()
      {
        return n;
      }

    public Pool2D pool(int index)
      {
        return pools[index];
      }

    public String name()
      {
        return layerName;
      }

    public double[] output()
      {
        return out;
      }

    /* Return this layer's output length */
    public int outputLen()
      {
        int ctr = 0;
        int i;

        for(i = 0; i < n; i++)
          ctr += (int)(Math.floor((double)(inputW - pools[i].w + 1) / (double)pools[i].stride_h) *
                       Math.floor((double)(inputH - pools[i].h + 1) / (double)pools[i].stride_v));

        return ctr;
      }

    /* Run the given input vector 'x' of length 'inputW' * 'inputH' through the Pool2DLayer.
       The understanding for this function is that pooling never runs off the edge of the input, and that there is
       only one "color-channel."
       Output is stored internally in 'out'. */
    public int run(double[] xvec)
      {
        int i = 0;                                                  //  Pool array iterator
        int o = 0;                                                  //  Output iterator
        int ctr;                                                    //  Only used in median pooling
        int x, y;                                                   //  2D input iterators
        int j, k;                                                   //  2D pool iterators

        double[] cache;                                             //  Intermediate buffer
        int cacheLen;                                               //  Length of that buffer
        boolean cacheLenEven = false;
        int index;                                                  //  Used in median pooling
        double val;

        for(i = 0; i < n; i++)                                      //  For each pool
          {
            switch(pools[i].f)                                      //  Prefer one "if" per layer to one "if" per iteration
              {
                case MAX_POOL:    for(y = 0; y <= inputH - pools[i].h; y += pools[i].stride_v)
                                    {
                                      for(x = 0; x <= inputW - pools[i].w; x += pools[i].stride_h)
                                        {
                                          val = Double.NEGATIVE_INFINITY;
                                          for(k = 0; k < pools[i].h; k++)
                                            {
                                              for(j = 0; j < pools[i].w; j++)
                                                {
                                                  if(xvec[(y + k) * inputW + x + j] > val)
                                                    val = xvec[(y + k) * inputW + x + j];
                                                }
                                            }
                                          out[o] = val;
                                          o++;
                                        }
                                    }
                                  break;
                case MIN_POOL:    for(y = 0; y <= inputH - pools[i].h; y += pools[i].stride_v)
                                    {
                                      for(x = 0; x <= inputW - pools[i].w; x += pools[i].stride_h)
                                        {
                                          val = Double.POSITIVE_INFINITY;
                                          for(k = 0; k < pools[i].h; k++)
                                            {
                                              for(j = 0; j < pools[i].w; j++)
                                                {
                                                  if(xvec[(y + k) * inputW + x + j] < val)
                                                    val = xvec[(y + k) * inputW + x + j];
                                                }
                                            }
                                          out[o] = val;
                                          o++;
                                        }
                                    }
                                  break;
                case AVG_POOL:    for(y = 0; y <= inputH - pools[i].h; y += pools[i].stride_v)
                                    {
                                      for(x = 0; x <= inputW - pools[i].w; x += pools[i].stride_h)
                                        {
                                          val = 0.0;
                                          for(k = 0; k < pools[i].h; k++)
                                            {
                                              for(j = 0; j < pools[i].w; j++)
                                                val += xvec[(y + k) * inputW + x + j];
                                            }
                                          out[o] = val / (pools[i].w * pools[i].h);
                                          o++;
                                        }
                                    }
                                  break;
                case MEDIAN_POOL: cacheLen = pools[i].w * pools[i].h;
                                  cache = new double[cacheLen];
                                  cacheLenEven = (cacheLen % 2 == 0);
                                  if(cacheLenEven)
                                    index = cacheLen / 2 - 1;
                                  else
                                    index = (cacheLen - 1) / 2;

                                  for(y = 0; y <= inputH - pools[i].h; y += pools[i].stride_v)
                                    {
                                      for(x = 0; x <= inputW - pools[i].w; x += pools[i].stride_h)
                                        {
                                          ctr = 0;
                                          for(k = 0; k < pools[i].h; k++)
                                            {
                                              for(j = 0; j < pools[i].w; j++)
                                                {
                                                  cache[ctr] = xvec[(y + k) * inputW + x + j];
                                                  ctr++;
                                                }
                                            }

                                          pooling_quicksort(true, cache, 0, cacheLen - 1);

                                          if(cacheLenEven)
                                            out[o] = 0.5 * (cache[index] + cache[index + 1]);
                                          else
                                            out[o] = cache[index];

                                          o++;
                                        }
                                    }
                                  cache = null;
                                  break;
              }
          }

        System.gc();                                                //  Call the garbage collector

        return outlen;
      }

    public boolean read(DataInputStream fp)
      {
        int ctr;
        int w, h, stride_h, stride_v, f;

        ByteBuffer byteBuffer;
        int allocation;
        byte byteArr[];

        allocation = 12 + NeuralNet.LAYER_NAME_LEN;                 //  Allocate space for 3 ints and the layer name
        byteArr = new byte[allocation];

        try
          {
            fp.read(byteArr);
          }
        catch(IOException ioErr)
          {
            System.out.println("ERROR: Unable to read 2D Pooling Layer header from file.");
            return false;
          }

        byteBuffer = ByteBuffer.allocate(allocation);
        byteBuffer = ByteBuffer.wrap(byteArr);
        byteBuffer.order(ByteOrder.LITTLE_ENDIAN);                  //  Read little-endian

        inputW = byteBuffer.getInt();                               //  (int) Read the input width from file
        inputH = byteBuffer.getInt();                               //  (int) Read the input height from file
        n = byteBuffer.getInt();                                    //  (int) Read the number of pools from file

        byteArr = new byte[NeuralNet.LAYER_NAME_LEN];               //  Allocate
        for(ctr = 0; ctr < NeuralNet.LAYER_NAME_LEN; ctr++)         //  Read into array
          byteArr[ctr] = byteBuffer.get();
        layerName = new String(byteArr, StandardCharsets.UTF_8);

        pools = new Pool2D[n];                                      //  Allocate pool array

        for(ctr = 0; ctr < n; ctr++)                                //  For each filter
          {
            allocation = 17;                                        //  Allocate space for 4 ints (4 bytes each) + 1 byte
            byteArr = new byte[allocation];

            try
              {
                fp.read(byteArr);
              }
            catch(IOException ioErr)
              {
                System.out.println("ERROR: Unable to read 2D pool from file.");
                return false;
              }

            byteBuffer = ByteBuffer.allocate(allocation);
            byteBuffer = ByteBuffer.wrap(byteArr);
            byteBuffer.order(ByteOrder.LITTLE_ENDIAN);              //  Read little-endian

            w = byteBuffer.getInt();
            h = byteBuffer.getInt();
            stride_h = byteBuffer.getInt();
            stride_v = byteBuffer.getInt();
            f = byteBuffer.get();

            pools[ctr] = new Pool2D(w, h);
            pools[ctr].stride_h = stride_h;
            pools[ctr].stride_v = stride_v;
            pools[ctr].f = (int)f;
          }

        outlen = outputLen();
        out = new double[outlen];

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

        allocation = 12 + NeuralNet.LAYER_NAME_LEN +                //  3 ints and the layer name,
                     17 * n;                                        //  plus 17 bytes per pooling (4 ints @ 4 bytes apiece + 1 byte).
        byteBuffer = ByteBuffer.allocate(allocation);
        byteBuffer.order(ByteOrder.LITTLE_ENDIAN);                  //  Write little-endian

        byteBuffer.putInt(inputW);                                  //  (int) Save Pooling Layer's input width to file
        byteBuffer.putInt(inputH);                                  //  (int) Save Pooling Layer's input height to file
        byteBuffer.putInt(n);                                       //  (int) Save number of Pooling Layer's pools to file

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

        for(ctr = 0; ctr < n; ctr++)                                //  For each pool...
          {
            byteBuffer.putInt(pools[ctr].w);                        //  (int) Save pool width to file
            byteBuffer.putInt(pools[ctr].h);                        //  (int) Save pool height to file
            byteBuffer.putInt(pools[ctr].stride_h);                 //  (int) Save pool horizontal stride to file
            byteBuffer.putInt(pools[ctr].stride_v);                 //  (int) Save pool vertical stride to file
            byteBuffer.put((byte)pools[ctr].f);                     //  (byte) Save pool function flag to file
          }

        byteArr = byteBuffer.array();

        try
          {
            fp.write(byteArr, 0, byteArr.length);
          }
        catch(IOException ioErr)
          {
            System.out.println("ERROR: Unable to write 2D Pooling Layer to file.");
            return false;
          }

        byteArr = null;                                             //  Release the array
        System.gc();                                                //  Call the garbage collector

        return true;
      }

    private void pooling_quicksort(boolean desc, double[] a, int lo, int hi)
      {
        int p;

        if(lo < hi)
          {
            p = pooling_partition(desc, a, lo, hi);

            if(p > 0)                                               //  PREVENT ROLL-OVER TO Integer.MAX_VALUE
              pooling_quicksort(desc, a, lo, p - 1);                //  Left side: start quicksort
            if(p < Integer.MAX_VALUE)                               //  PREVENT ROLL-OVER TO Integer.MIN_VALUE
              pooling_quicksort(desc, a, p + 1, hi);                //  Right side: start quicksort
          }

        return;
      }

    private int pooling_partition(boolean desc, double[] a, int lo, int hi)
      {
        double pivot = a[hi];
        int i = lo;
        int j;
        double tmpFloat;
        boolean trigger;

        for(j = lo; j < hi; j++)
          {
            if(desc)
              trigger = (a[j] > pivot);                             //  SORT DESCENDING
            else
              trigger = (a[j] < pivot);                             //  SORT ASCENDING

            if(trigger)
              {
                tmpFloat = a[i];                                    //  Swap a[i] with a[j]
                a[i]   = a[j];
                a[j]   = tmpFloat;

                i++;
              }
          }

        tmpFloat = a[i];                                            //  Swap a[i] with a[hi]
        a[i]  = a[hi];
        a[hi] = tmpFloat;

        return i;
      }

    public class Pool2D
      {
        public int w;                                               //  Width of the filter
        public int h;                                               //  Height of the filter

        public int stride_h;                                        //  Left-right stride
        public int stride_v;                                        //  Top-bottom stride

        public int f;                                               //  Function flag

        public Pool2D(int width, int height)
          {
            w = width;
            h = height;

            stride_h = 1;                                           //  Default to left-right stride = 1
            stride_v = 1;                                           //  Default to top-bottom stride = 1

            f = MAX_POOL;                                           //  Default to max pooling
          }
      }
  }
