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

public class Conv2DLayer
  {
    private int inputW, inputH;                                     //  Dimensions of the input
    private int n;                                                  //  Number of processing units in this layer =
                                                                    //  number of filters in this layer
    private Filter2D filters[];                                     //  Array of 2D filter structs

    private double out[];                                           //  Output buffer
    private int outlen;                                             //  Length of the output buffer

    private String name;                                            //  Name of this layer

    /*  Conv2DLayer 'nameStr' shall receive (w x h) input. */
    public Conv2DLayer(int w, int h, String nameStr)
      {
        inputW = w;                                                 //  Set this layer's input dimensions
        inputH = h;
        n = 0;                                                      //  New layer initially contains zero filters
        outlen = 0;
        name = nameStr;
      }

    /*  New Conv2DLayer shall receive (w x h) input. */
    public Conv2DLayer(int w, int h)
      {
        this(w, h, "");
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
        name = nameStr;
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

    /* Return the layer's output length */
    public int outputLen()
      {
        int ctr = 0;
        int i;

        for(i = 0; i < n; i++)
          ctr += (int)(Math.floor((double)(inputW - filters[i].w + 1) / (double)filters[i].stride_h) *
                       Math.floor((double)(inputH - filters[i].h + 1) / (double)filters[i].stride_v));

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
            filterOutputLen = (int)(Math.floor((double)(inputW - filters[i].w + 1) / (double)filters[i].stride_h) *
                                    Math.floor((double)(inputH - filters[i].h + 1) / (double)filters[i].stride_v));
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

    private class Filter2D
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
