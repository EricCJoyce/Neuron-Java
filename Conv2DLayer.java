/**************************************************************************************************
 Neural Network library, by Eric C. Joyce

 Model a Convolutional Layer as an array of one or more 2D filters:

  input mat{X} w, h       filter1      activation function vector f
 [ x11 x12 x13 x14 ]    [ w11 w12 ]   [ func1 func2 ]
 [ x21 x22 x23 x24 ]    [ w21 w22 ]
 [ x31 x32 x33 x34 ]    [ bias ]       auxiliary vector alpha
 [ x41 x42 x43 x44 ]                  [ param1 param2 ]
 [ x51 x52 x53 x54 ]      filter2
                      [ w11 w12 w13 ]
                      [ w21 w22 w23 ]
                      [ w31 w32 w33 ]
                      [ bias ]

 Filters needn't be arranged from smallest to largest; this is just for illustration.

 Note that this file does NOT seed the randomizer. That should be done by the parent program.
***************************************************************************************************/

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;

public class Conv2DLayer
  {
    private int inputW, inputH;                                     //  Dimensions of the input
    private int n;                                                  //  Number of processing units in this layer =
                                                                    //  number of filters in this layer
    private Filter2D filters[];                                     //  Array of 2D filter structs

    private double out[];                                           //  Output buffer
    private int outlen;                                             //  Length of the output buffer

    private String layerName;                                       //  Name of this layer

    /*  Conv2DLayer 'nameStr' shall receive (w x h) input. */
    public Conv2DLayer(int w, int h, String nameStr)
      {
        inputW = w;                                                 //  Set this layer's input dimensions
        inputH = h;
        n = 0;                                                      //  New layer initially contains zero filters
        outlen = 0;
        layerName = nameStr;
      }

    /*  New Conv2DLayer shall receive (w x h) input. */
    public Conv2DLayer(int w, int h)
      {
        this(w, h, "");
      }

    /* Use placeholder arguments */
    public Conv2DLayer()
      {
        this(1, 1, "");
      }

    /* Add a Filter2D to an existing Conv2DLayer.
       The new filter shall have dimensions 'filterW' by 'filterH'. */
    public int addFilter(int filterW, int filterH)
      {
        int i;
        int ctr;

        Filter2D tmp_filters[];

        if(n == 0)
          {
            filters = new Filter2D[1];                              //  Allocate filter in 'filters' array
            filters[0] = new Filter2D(filterW, filterH);
          }
        else
          {
            tmp_filters = new Filter2D[n];                          //  Allocate temporary arrays
            System.arraycopy(filters, 0, tmp_filters, 0, n);        //  Copy to temporary arrays

            filters = new Filter2D[n + 1];                          //  Re-allocate 'filters' array
            System.arraycopy(tmp_filters, 0, filters, 0, n);        //  Copy back into (expanded) original arrays

            filters[n] = new Filter2D(filterW, filterH);            //  Allocate another filter in 'filters' array

            tmp_filters = null;                                     //  Force release of allocated memory
            System.gc();                                            //  Call the garbage collector
          }

        outlen = outputLen();                                       //  Re-compute the output count
        out = new double[outlen];                                   //  Re-allocate this layer's output array

        n++;                                                        //  Increment the number of filters/units

        return n;
      }

    /* Set entirety of i-th filter; w is length width * height + 1.
       Input array 'w' is expected to be ROW-MAJOR:
            filter
       [ w0  w1  w2  ]
       [ w3  w4  w5  ]
       [ w6  w7  w8  ]  [ bias (w9) ]  */
    public void setW_i(double[] w, int i)
      {
        int ctr;

        if(i < filters.length)
          {
            for(ctr = 0; ctr < filters[i].w * filters[i].h + 1; ctr++)
              filters[i].W[ctr] = w[ctr];
          }

        return;
      }

    /* Set filter[i], weight[j] of the given layer */
    public void setW_ij(double w, int i, int j)
      {
        if(i < n && j < filters[i].w * filters[i].h + 1)
          filters[i].W[j] = w;
      }

    /* Set filter[i]'s horizontal stride for the given layer */
    public void setHorzStride_i(int stride, int i)
      {
        if(i < n)
          {
            filters[i].stride_h = stride;

            outlen = outputLen();                                   //  Re-compute layer's output length and reallocate its output buffer
            out = new double[outlen];                               //  Re-allocate this layer's output array
          }

        return;
      }

    /* Set filter[i]'s vertical stride for the given layer */
    public void setVertStride_i(int stride, int i)
      {
        if(i < n)
          {
            filters[i].stride_v = stride;

            outlen = outputLen();                                   //  Re-compute layer's output length and reallocate its output buffer
            out = new double[outlen];                               //  Re-allocate this layer's output array
          }
        return;
      }

    /* Set the activation function for unit[i] of the given layer */
    public void setF_i(int func, int index)
      {
        if(index < n)
          filters[index].f = func;
        return;
      }

    /* Set the activation function parameter for unit[i] of the given layer */
    public void setA_i(double a, int index)
      {
        if(index < n)
          filters[index].alpha = a;
        return;
      }

    /* Set the name of the given Convolutional Layer */
    public void setName(String nameStr)
      {
        layerName = nameStr;
        return;
      }

    /* Print the details of the given Conv2DLayer 'layer' */
    public void print()
      {
        int i, x, y;

        for(i = 0; i < n; i++)                                      //  Draw each filter
          {
            System.out.printf("Filter %d\n", i);
            for(y = 0; y < filters[i].h; y++)
              {
                System.out.print("  [");
                for(x = 0; x < filters[i].w; x++)
                  {
                    if(filters[i].W[y * filters[i].w + x] >= 0.0)
                      System.out.printf(" %.5f ", filters[i].W[y * filters[i].w + x]);
                    else
                      System.out.printf("%.5f ", filters[i].W[y * filters[i].w + x]);
                  }
                System.out.print("]\n");
              }
            System.out.print("  Func:  ");
            switch(filters[i].f)
              {
                case ActivationFunction.RELU:                System.out.print("ReLU   ");  break;
                case ActivationFunction.LEAKY_RELU:          System.out.print("L.ReLU ");  break;
                case ActivationFunction.SIGMOID:             System.out.print("Sig.   ");  break;
                case ActivationFunction.HYPERBOLIC_TANGENT:  System.out.print("tanH   ");  break;
                case ActivationFunction.SOFTMAX:             System.out.print("SoftMx ");  break;
                case ActivationFunction.SYMMETRICAL_SIGMOID: System.out.print("SymSig ");  break;
                case ActivationFunction.THRESHOLD:           System.out.print("Thresh ");  break;
                case ActivationFunction.LINEAR:              System.out.print("Linear ");  break;
              }
            System.out.print("\n");
            System.out.printf("  Param: %.5f\n", filters[i].alpha);
            System.out.printf("  Bias:  %.5f\n", filters[i].W[filters[i].h * filters[i].w]);
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

    public int numFilters()
      {
        return n;
      }

    public Filter2D filter(int index)
      {
        return filters[index];
      }

    public String name()
      {
        return layerName;
      }

    public double[] output()
      {
        return out;
      }

    /* Return the layer's output length */
    public int outputLen()
      {
        int ctr = 0;
        int i;

        for(i = 0; i < n; i++)
          ctr += (int)( (Math.floor((double)(inputW - filters[i].w) / (double)filters[i].stride_h) + 1.0) *
                        (Math.floor((double)(inputH - filters[i].h) / (double)filters[i].stride_v) + 1.0) );

        return ctr;
      }

    /* Run the given input vector 'xvec' of length 'inputW' * 'inputH' through the Conv2DLayer.
       The understanding for this function is that convolution never runs off the edge of the input,
       and that there is only one "color-channel."
       Output is stored internally in 'out'. */
    public int run(double[] xvec)
      {
        int i, o = 0, c;                                            //  Iterators for the filters, the output vector, and the cache
        int s;                                                      //  Cache iterator
        int x, y;                                                   //  Input iterators
        int j, k;                                                   //  Filter iterators
        int filterOutputLen = 0;                                    //  Length of a single filter's output vector
        double[] cache;                                             //  Output array for a single filter
        double softmaxdenom;
        double val;

        for(i = 0; i < n; i++)                                      //  For each filter
          {
            c = 0;
            softmaxdenom = 0.0;
            filterOutputLen = (int)( (Math.floor((double)(inputW - filters[i].w) / (double)filters[i].stride_h) + 1.0) *
                                     (Math.floor((double)(inputH - filters[i].h) / (double)filters[i].stride_v) + 1.0) );
            cache = new double[filterOutputLen];

            for(y = 0; y <= inputH - filters[i].h; y += filters[i].stride_v)
              {
                for(x = 0; x <= inputW - filters[i].w; x += filters[i].stride_h)
                  {
                    val = 0.0;
                    for(k = 0; k < filters[i].h; k++)
                      {
                        for(j = 0; j < filters[i].w; j++)
                          val += filters[i].W[k * filters[i].w + j] * xvec[(y + k) * inputW + x + j];
                      }
                                                                    //  Add bias
                    val += filters[i].W[filters[i].w * filters[i].h];
                    cache[c] = val;                                 //  Add the value to the cache
                    c++;
                  }
              }

            for(s = 0; s < c; s++)                                  //  In case one of the units is a softmax unit,
              softmaxdenom += Math.pow(Math.E, cache[s]);           //  compute all exp()'s so we can sum them.

            for(s = 0; s < c; s++)
              {
                switch(filters[i].f)
                  {
                    case ActivationFunction.RELU:                out[o] = ActivationFunction.relu(cache[s]);  break;
                    case ActivationFunction.LEAKY_RELU:          out[o] = ActivationFunction.leaky_relu(cache[s], filters[i].alpha);  break;
                    case ActivationFunction.SIGMOID:             out[o] = ActivationFunction.sigmoid(cache[s], filters[i].alpha);  break;
                    case ActivationFunction.HYPERBOLIC_TANGENT:  out[o] = ActivationFunction.tanh(cache[s], filters[i].alpha);  break;
                    case ActivationFunction.SOFTMAX:             out[o] = ActivationFunction.softmax(cache[s], softmaxdenom);  break;
                    case ActivationFunction.SYMMETRICAL_SIGMOID: out[o] = ActivationFunction.sym_sigmoid(cache[s], filters[i].alpha);  break;
                    case ActivationFunction.THRESHOLD:           out[o] = ActivationFunction.threshold(cache[s], filters[i].alpha);  break;
                    default:                                     out[o] = ActivationFunction.linear(cache[s], filters[i].alpha);
                  }
                o++;
              }

            cache = null;                                           //  Release the cache for this filter
            System.gc();                                            //  Call the garbage collector
          }

        return outlen;
      }

    public boolean read(DataInputStream fp)
      {
        int ctr, wctr;
        int w, h, stride_h, stride_v, f;
        double alpha;

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
            System.out.println("ERROR: Unable to read 2D Convolutional Layer header from file.");
            return false;
          }

        byteBuffer = ByteBuffer.allocate(allocation);
        byteBuffer = ByteBuffer.wrap(byteArr);
        byteBuffer.order(ByteOrder.LITTLE_ENDIAN);                  //  Read little-endian

        inputW = byteBuffer.getInt();                               //  (int) Read the input width from file
        inputH = byteBuffer.getInt();                               //  (int) Read the input height from file
        n = byteBuffer.getInt();                                    //  (int) Read the number of filters from file

        byteArr = new byte[NeuralNet.LAYER_NAME_LEN];               //  Allocate
        for(ctr = 0; ctr < NeuralNet.LAYER_NAME_LEN; ctr++)         //  Read into array
          byteArr[ctr] = byteBuffer.get();
        layerName = new String(byteArr, StandardCharsets.UTF_8);

        filters = new Filter2D[n];                                  //  Allocate filter array

        for(ctr = 0; ctr < n; ctr++)                                //  For each filter
          {
            allocation = 25;                                        //  Allocate space for 4 ints (4 bytes each) + 1 byte + 1 double (8 bytes)
            byteArr = new byte[allocation];

            try
              {
                fp.read(byteArr);
              }
            catch(IOException ioErr)
              {
                System.out.println("ERROR: Unable to read 2D convolutional filter header from file.");
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
            alpha = byteBuffer.getDouble();

            filters[ctr] = new Filter2D(w, h);
            filters[ctr].stride_h = stride_h;
            filters[ctr].stride_v = stride_v;
            filters[ctr].f = (int)f;
            filters[ctr].alpha = alpha;

            filters[ctr].W = new double[w * h + 1];                 //  Allocate array of filter weights

            allocation = 8 * (w * h + 1);                           //  Allocate space to read weights
            byteArr = new byte[allocation];

            try
              {
                fp.read(byteArr);
              }
            catch(IOException ioErr)
              {
                System.out.println("ERROR: Unable to read 2D convolutional filter weights from file.");
                return false;
              }

            byteBuffer = ByteBuffer.allocate(allocation);
            byteBuffer = ByteBuffer.wrap(byteArr);
            byteBuffer.order(ByteOrder.LITTLE_ENDIAN);              //  Read little-endian

            for(wctr = 0; wctr < w * h + 1; wctr++)
              filters[ctr].W[wctr] = byteBuffer.getDouble();
          }

        outlen = outputLen();
        out = new double[outlen];

        byteArr = null;                                             //  Release the array
        System.gc();                                                //  Call the garbage collector

        return true;
      }

    public boolean write(DataOutputStream fp)
      {
        int ctr, wctr;

        ByteBuffer byteBuffer;
        int allocation;
        byte byteArr[];
                                                                    //  Allocate space for
        allocation = 12 + NeuralNet.LAYER_NAME_LEN;                 //  3 ints and the layer name.
        for(ctr = 0; ctr < n; ctr++)                                //  Add, for each filter,
          {
            allocation += 25;                                       //  4 ints (4 bytes each) + 1 byte + 1 double (8 bytes),
            allocation += (filters[ctr].w * filters[ctr].h + 1) * 8;//  w*h+1 doubles (8 bytes each).
          }
        byteBuffer = ByteBuffer.allocate(allocation);
        byteBuffer.order(ByteOrder.LITTLE_ENDIAN);                  //  Write little-endian

        byteBuffer.putInt(inputW);                                  //  (int) Save Conv2DLayer input width to file
        byteBuffer.putInt(inputH);                                  //  (int) Save Conv2DLayer input height to file
        byteBuffer.putInt(n);                                       //  (int) Save Conv2DLayer's number of filters to file

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

        for(ctr = 0; ctr < n; ctr++)                                //  For each filter...
          {
            byteBuffer.putInt(filters[ctr].w);                      //  (int) Save filter width to file
            byteBuffer.putInt(filters[ctr].h);                      //  (int) Save filter height to file
            byteBuffer.putInt(filters[ctr].stride_h);               //  (int) Save filter horizontal stride to file
            byteBuffer.putInt(filters[ctr].stride_v);               //  (int) Save filter vertical stride to file
            byteBuffer.put((byte)filters[ctr].f);                   //  (byte) Save filter activation function flag to file
            byteBuffer.putDouble(filters[ctr].alpha);               //  (byte) Save filter function parameter to file
            for(wctr = 0; wctr < filters[ctr].w * filters[ctr].h + 1; wctr++)
              byteBuffer.putDouble(filters[ctr].W[wctr]);           //  (double) Save filter weights to file
          }

        byteArr = byteBuffer.array();

        try
          {
            fp.write(byteArr, 0, byteArr.length);
          }
        catch(IOException ioErr)
          {
            System.out.println("ERROR: Unable to write 2D Convolutional Layer to file.");
            return false;
          }

        byteArr = null;                                             //  Release the array
        System.gc();                                                //  Call the garbage collector

        return true;
      }

    public class Filter2D
      {
        public int w;                                               //  Width of the filter
        public int h;                                               //  Height of the filter

        public int stride_h;                                        //  Left-right stride
        public int stride_v;                                        //  Top-bottom stride

        public int f;                                               //  Activation function flag
        public double alpha;                                        //  Activation function parameter

        public double W[];                                          //  Array of (w * h) weights, arranged row-major, +1 for the bias

        public Filter2D(int filterW, int filterH)
          {
            int ctr;

            w = filterW;
            h = filterH;

            stride_h = 1;                                           //  Default to left-right stride = 1
            stride_v = 1;                                           //  Default to top-bottom stride = 1

            f = ActivationFunction.RELU;                            //  Default to ReLU
            alpha = 1.0;                                            //  Default to 1.0

            W = new double[w * h + 1];

            for(ctr = 0; ctr < w * h + 1; ctr++)
              W[ctr] = -1.0 + Math.random() * 2.0;                  //  Generate random numbers in [ -1.0, 1.0 ]
          }
      }
  }
