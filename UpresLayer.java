/**************************************************************************************************
 Neural Network library, by Eric C. Joyce

 An upres layer serves to prepare input for (transposed) convolution.
  s = stride
  p = padding

    input mat{X}         output for s = 2, p = 0        output for s = 2, p = 1
 [ x11 x12 x13 x14 ]    [ x11 0 x12 0 x13 0 x14 ]    [ 0  0  0  0  0  0  0  0  0 ]
 [ x21 x22 x23 x24 ]    [  0  0  0  0  0  0  0  ]    [ 0 x11 0 x12 0 x13 0 x14 0 ]
 [ x31 x32 x33 x34 ]    [ x21 0 x22 0 x23 0 x24 ]    [ 0  0  0  0  0  0  0  0  0 ]
 [ x41 x42 x43 x44 ]    [  0  0  0  0  0  0  0  ]    [ 0 x21 0 x22 0 x23 0 x24 0 ]
 [ x51 x52 x53 x54 ]    [ x31 0 x32 0 x33 0 x34 ]    [ 0  0  0  0  0  0  0  0  0 ]
                        [  0  0  0  0  0  0  0  ]    [ 0 x31 0 x32 0 x33 0 x34 0 ]
                        [ x41 0 x42 0 x43 0 x44 ]    [ 0  0  0  0  0  0  0  0  0 ]
                        [  0  0  0  0  0  0  0  ]    [ 0 x41 0 x42 0 x43 0 x44 0 ]
                        [ x51 0 x52 0 x53 0 x54 ]    [ 0  0  0  0  0  0  0  0  0 ]
                                                     [ 0 x51 0 x52 0 x53 0 x54 0 ]
                                                     [ 0  0  0  0  0  0  0  0  0 ]

 Note that this file does NOT seed the randomizer. That should be done by the parent program.
***************************************************************************************************/

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.nio.charset.StandardCharsets;

public class UpresLayer
  {
    public static final int FILL_ZERO   = 0;                        //  Fill strides or pad using zeroes
    public static final int FILL_SAME   = 1;                        //  Fill strides or pad using duplicates of the nearest value
    public static final int FILL_INTERP = 2;                        //  Fill strides or pad using bilinear interpolation

    private int inputW, inputH;                                     //  Dimensions of the input
    private UpresParams params[];                                   //  Array of Up-resolution parameters structures
    private int n;                                                  //  Length of that array

    private double out[];
    private int outlen;                                             //  Length of the output buffer
    private String layerName;

    public UpresLayer(int w, int h, String nameStr)
      {
        inputW = w;
        inputH = h;
        n = 0;                                                      //  Initially, no up-ressings
        outlen = 0;                                                 //  And therefore, no output
        layerName = nameStr;
      }

    public UpresLayer(int w, int h)
      {
        this(w, h, "");
      }

    /* Add an "up-ressing" to an existing UpresLayer.
       The new "up-ressing" shall have stride 'stride' and padding 'padding'. */
    public int addParams(int stride, int padding)
      {
        int i;
        int ctr;

        UpresParams tmp_params[];

        if(n == 0)
          {
            params = new UpresParams[1];                            //  Allocate an upressing in 'params' array
            params[0] = new UpresParams(stride, padding);
          }
        else
          {
            tmp_params = new UpresParams[n];                        //  Allocate temporary arrays
            System.arraycopy(params, 0, tmp_params, 0, n);          //  Copy to temporary arrays

            params = new UpresParams[n + 1];                        //  Re-allocate 'params' array
            System.arraycopy(tmp_params, 0, params, 0, n);          //  Copy back into (expanded) original arrays

            params[n] = new UpresParams(stride, padding);           //  Allocate another filter in 'params' array

            tmp_params = null;                                      //  Force release of allocated memory
            System.gc();                                            //  Call the garbage collector
          }

        outlen = outputLen();                                       //  Re-compute the output count
        out = new double[outlen];                                   //  Re-allocate this layer's output array

        n++;                                                        //  Increment the number of filters/units

        return n;
      }

    public void setHorzStride(int stride, int i)
      {
        if(i < n && stride > 0)
          {
            params[i].stride_h = stride;

            outlen = outputLen();                                   //  Re-compute the output count
            out = new double[outlen];                               //  Re-allocate this layer's output array
          }
        return;
      }

    public void setVertStride(int stride, int i)
      {
        if(i < n && stride > 0)
          {
            params[i].stride_v = stride;

            outlen = outputLen();                                   //  Re-compute the output count
            out = new double[outlen];                               //  Re-allocate this layer's output array
          }
        return;
      }

    public void setHorzPad(int pad, int i)
      {
        if(i < n && pad >= 0)
          {
            params[i].padding_h = pad;

            outlen = outputLen();                                   //  Re-compute the output count
            out = new double[outlen];                               //  Re-allocate this layer's output array
          }
        return;
      }

    public void setVertPad(int pad, int i)
      {
        if(i < n && pad >= 0)
          {
            params[i].padding_v = pad;

            outlen = outputLen();                                   //  Re-compute the output count
            out = new double[outlen];                               //  Re-allocate this layer's output array
          }
        return;
      }

    public void setStrideMethod(int strideMethod, int i)
      {
        if(i < n && strideMethod >= FILL_ZERO && strideMethod <= FILL_INTERP)
          params[i].sMethod = strideMethod;
        return;
      }

    public void setPaddingMethod(char padMethod, int i)
      {
        if(i < n && padMethod >= FILL_ZERO && padMethod <= FILL_INTERP)
          params[i].pMethod = padMethod;
        return;
      }

    /* Set the name of the given Upres Layer */
    public void setName(String nameStr)
      {
        layerName = nameStr;
        return;
      }

    /* Print the details of the given UpresLayer 'layer' */
    public void print()
      {
        int i;

        System.out.printf("Input Shape = (%d, %d)\n", inputW, inputH);

        for(i = 0; i < n; i++)                                      //  Draw each up-ressing
          {
            System.out.printf("Parameters %d\n", i);
            System.out.printf("  H.stride  = %d\n", params[i].stride_h);
            System.out.printf("  V.stride  = %d\n", params[i].stride_v);
            System.out.printf("  H.padding = %d\n", params[i].padding_h);
            System.out.printf("  V.padding = %d\n", params[i].padding_v);
            System.out.printf("  Stride    = ");
            switch(params[i].sMethod)
              {
                case FILL_ZERO:    System.out.print("zero");         break;
                case FILL_SAME:    System.out.print("same");         break;
                case FILL_INTERP:  System.out.print("interpolate");  break;
              }
            System.out.print("\n  Padding   = ");
            switch(params[i].pMethod)
              {
                case FILL_ZERO:    System.out.printf("zero");         break;
                case FILL_SAME:    System.out.printf("same");         break;
                case FILL_INTERP:  System.out.printf("interpolate");  break;
              }
            System.out.print("\n");
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

    public int numUpres()
      {
        return n;
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
        int i;
        int ctr = 0;

        for(i = 0; i < n; i++)
          ctr += (inputW * (params[i].stride_h + 1) - params[i].stride_h + params[i].padding_h + params[i].padding_h) *
                 (inputH * (params[i].stride_v + 1) - params[i].stride_v + params[i].padding_v + params[i].padding_v);

        return ctr;
      }

    /* Run the given input vector 'xvec' of length 'inputW' * 'inputH' through the UpresLayer.
       The understanding for this function is that there is only one "color-channel."
       Output is stored internally in layer->out. */
    public int run(double[] xvec)
      {
        int i;                                                      //  Up-ressing iterator
        int o = 0;                                                  //  Output iterator
        int x, y;                                                   //  Iterators
        int x_src, y_src;                                           //  Used in zero-fill stride to iterate over source
        int cache_w, cache_h;                                       //  Dimensions of the inner-rectangle (without padding applied yet)
        int output_w, output_h;                                     //  Dimensions per up-ressing of the padded output
        double x_prime, y_prime;                                    //  Inter-pixel locations in source
        double a, b;                                                //  Fractional parts of clipped doubles
        double sc_inv_h, sc_inv_v;                                  //  Scaling factors of the inverse transformation
        double val;                                                 //  Stores and compares neighboring pixel influence
        double cache[];                                             //  The "inner rectangle" we compute first
        int ctr;

        for(ctr = 0; ctr < outlen; ctr++)                           //  Blank out the output buffer
          out[ctr] = 0.0;

        for(i = 0; i < n; i++)                                      //  For each up-ressing, write the inner rectangle to cache and then wreath with padding.
          {
                                                                    //  Compute the shape of the inner rectangle
            cache_w = inputW * (params[i].stride_h + 1) - params[i].stride_h;
            cache_h = inputH * (params[i].stride_v + 1) - params[i].stride_v;

            output_w = cache_w + 2 * params[i].padding_h;           //  Compute the shape of the padded rectangle
            output_h = cache_h + 2 * params[i].padding_v;
                                                                    //  Allocate cache for the inner rectangle
            cache = new double[cache_w * cache_h];

            ctr = 0;                                                //  Reset counter: this now acts as our temporary output iterator

            if(params[i].sMethod == FILL_INTERP)                    //  Fill strides using bilinear interpolation
              {
                sc_inv_h = (double)inputW / (double)(cache_w);
                sc_inv_v = (double)inputH / (double)(cache_h);

                for(y = 0; y < cache_h; y++)
                  {
                    for(x = 0; x < cache_w; x++)
                      {
                        x_prime = (double)x * sc_inv_h;             //  Where in the source does this pixel fall?
                        y_prime = (double)y * sc_inv_v;

                        a = x_prime - (double)((int)x_prime);       //  Clip the fractional parts, store them in a and b:
                        b = y_prime - (double)((int)y_prime);       //  weigh the influences of neighboring pixels.

                        cache[ctr] = ((1.0 - a) * (1.0 - b)) * xvec[ (int)y_prime      * inputW + (int)x_prime    ] +
                                     ((1.0 - a) * b)         * xvec[((int)y_prime + 1) * inputW + (int)x_prime    ] +
                                     (a * (1.0 - b))         * xvec[ (int)y_prime      * inputW + (int)x_prime + 1] +
                                     (a * b)                 * xvec[((int)y_prime + 1) * inputW + (int)x_prime + 1];

                        ctr++;
                      }
                  }
              }
            else if(params[i].sMethod == FILL_SAME)                 //  Fill strides in by duplicating the nearest source element
              {
                sc_inv_h = (double)inputW / (double)(cache_w);
                sc_inv_v = (double)inputH / (double)(cache_h);

                for(y = 0; y < cache_h; y++)
                  {
                    for(x = 0; x < cache_w; x++)
                      {
                        x_prime = (double)x * sc_inv_h;             //  Where in the source does this pixel fall?
                        y_prime = (double)y * sc_inv_v;

                        a = x_prime - (double)((int)x_prime);       //  Clip the fractional parts, store them in a and b:
                        b = y_prime - (double)((int)y_prime);       //  weigh the influences of neighboring pixels.

                        val = ((1.0 - a) * (1.0 - b));              //  Initial assumption: this pixel is nearest
                        cache[ctr]     = xvec[ (int)y_prime      * inputW + (int)x_prime    ];

                        if(((1.0 - a) * b) > val)                   //  Does this pixel have greater influence?
                          {
                            val = ((1.0 - a) * b);
                            cache[ctr] = xvec[((int)y_prime + 1) * inputW + (int)x_prime    ];
                          }
                        if((a * (1.0 - b)) > val)                   //  Does this pixel have greater influence?
                          {
                            val = (a * (1.0 - b));
                            cache[ctr] = xvec[ (int)y_prime      * inputW + (int)x_prime + 1];
                          }
                        if((a * b) > val)                           //  Does this pixel have greater influence?
                          {                                         //  (No point storing 'val' anymore.)
                            cache[ctr] = xvec[((int)y_prime + 1) * inputW + (int)x_prime + 1];
                          }

                        ctr++;
                      }
                  }
              }
            else                                                    //  Fill strides in with zeroes
              {
                x_src = 0;                                          //  Initialize source-iterators
                y_src = 0;

                for(y = 0; y < cache_h; y += (params[i].stride_v + 1))
                  {
                    for(x = 0; x < cache_w; x += (params[i].stride_h + 1))
                      {
                                                                    //  Copy source pixel
                        cache[ctr] = xvec[y_src * inputW + x_src];
                        x_src++;                                    //  Increment source x-iterator
                        ctr += (params[i].stride_h + 1);            //  Advance output-iterator by horizontal stride
                      }
                    x_src = 0;                                      //  Reset source x-iterator
                    y_src++;                                        //  Increment source y-iterator
                    ctr += (params[i].stride_v + 1) * cache_w;      //  Advance output-iterator by vertical stride
                  }
              }

            if(params[i].pMethod != FILL_ZERO)                      //  Duplicate extrema
              {
                                                                    //  First fill in the sides
                for(y = params[i].stride_v; y <= output_h - params[i].stride_v; y++)
                  {
                    for(x = 0; x < params[i].stride_h; x++)         //  Duplicate left side
                      out[o + output_w * y + x] = cache[(y - params[i].stride_v) * cache_w];
                                                                    //  Duplicate right side
                    for(x = params[i].stride_h + cache_w; x < output_w; x++)
                      out[o + output_w * y + params[i].stride_h + cache_w + x] = cache[(y - params[i].stride_v) * cache_w + cache_w - 1];
                  }
                                                                    //  Then fill the top and bottom
                for(y = 0; y < params[i].stride_v; y++)             //  Fill top by referring to the first side-padded row
                  {
                    for(x = 0; x < output_w; x++)
                      out[o + y * output_w + x] = out[o + params[i].stride_v * output_w + x];
                  }
                                                                    //  Fill bottom by referring to the last side-padded row
                for(y = params[i].stride_v + cache_h + 1; y < output_h; y++)
                  {
                    for(x = 0; x < output_w; x++)
                      out[o + y * output_w + x] = out[o + (params[i].stride_v + cache_h) * output_w + x];
                  }
              }
                                                                    //  Now, whether we had fancy padding or not, set cache into output buffer
            x_src = 0;                                              //  Reset; these now iterate over the cached inner rectangle
            y_src = 0;
            for(y = 0; y < output_h; y++)                           //  For every row in the padded output for the current up-ressing
              {                                                     //  if we have passed the topmost padding and not yet reached the bottommost
                if(y >= params[i].stride_v && y < output_h - params[i].stride_v)
                  {
                    for(x = 0; x < output_w; x++)                   //  For every column in the padded output for the current up-ressing
                      {                                             //  if we have passed the leftmost padding and not yet reached the rightmost
                        if(x >= params[i].stride_h && x < output_w - params[i].stride_h)
                          {
                                                                    //  Copy from cache
                            out[o] = cache[y_src * cache_w + x_src];
                            x_src++;                                //  Increment cache's x-iterator
                          }
                        o++;                                        //  Increment output buffer iterator
                      }
                    x_src = 0;                                      //  Reset cache's x-iterator
                    y_src++;                                        //  Increment cache's y-iterator
                  }
                else                                                //  Otherwise, skip a whole output row
                  o += output_w;
              }

            cache = null;                                           //  Release
            System.gc();                                            //  Call the garbage collector
          }

        return outlen;
      }

    public boolean read(DataInputStream fp)
      {
        int ctr;
        int stride_h, stride_v;
        int padding_h, padding_v;
        int sMethod, pMethod;
        byte buffer[];

        try
          {
            inputW = fp.readInt();                                  //  (int) Read layer input width from file
          }
        catch(IOException ioErr)
          {
            System.out.println("ERROR: Unable to read Upres Layer input width.");
            return false;
          }

        try
          {
            inputH = fp.readInt();                                  //  (int) Read layer input height from file
          }
        catch(IOException ioErr)
          {
            System.out.println("ERROR: Unable to read Upres Layer input height.");
            return false;
          }

        try
          {
            n = fp.readInt();                                       //  (int) Read number of layer pools from file
          }
        catch(IOException ioErr)
          {
            System.out.println("ERROR: Unable to read number of Upres Layer parameter tuples.");
            return false;
          }

        params = new UpresParams[n];                                //  Allocate

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
                System.out.println("ERROR: Unable to read Upres Layer name.");
                return false;
              }
          }
        layerName = new String(buffer, StandardCharsets.UTF_8);     //  Convert byte array to String
        buffer = null;                                              //  Release the array

        for(ctr = 0; ctr < n; ctr++)                                //  For each filter
          {
            try
              {
                stride_h = fp.readInt();
              }
            catch(IOException ioErr)
              {
                System.out.println("ERROR: Unable to read up-resolution horizontal stride from file.");
                return false;
              }
            try
              {
                stride_v = fp.readInt();
              }
            catch(IOException ioErr)
              {
                System.out.println("ERROR: Unable to read up-resolution vertical stride from file.");
                return false;
              }
            try
              {
                padding_h = fp.readInt();
              }
            catch(IOException ioErr)
              {
                System.out.println("ERROR: Unable to read up-resolution horizontal padding from file.");
                return false;
              }
            try
              {
                padding_v = fp.readInt();
              }
            catch(IOException ioErr)
              {
                System.out.println("ERROR: Unable to read up-resolution vertical padding from file.");
                return false;
              }
            try
              {
                sMethod = (int)fp.readByte();
              }
            catch(IOException ioErr)
              {
                System.out.println("ERROR: Unable to read up-resolution stride-method flag from file.");
                return false;
              }
            try
              {
                pMethod = (int)fp.readByte();
              }
            catch(IOException ioErr)
              {
                System.out.println("ERROR: Unable to read up-resolution padding-method flag from file.");
                return false;
              }
          }

        outlen = outputLen();
        out = new double[outlen];

        System.gc();                                                //  Call the garbage collector

        return true;
      }

    public boolean write(DataOutputStream fp)
      {
        int ctr;
        byte buffer[];

        try
          {
            fp.writeInt(inputW);                                    //  (int) Write layer input width to file
          }
        catch(IOException ioErr)
          {
            System.out.println("ERROR: Unable to write Upres Layer input width.");
            return false;
          }

        try
          {
            fp.writeInt(inputH);                                    //  (int) Write layer input height from file
          }
        catch(IOException ioErr)
          {
            System.out.println("ERROR: Unable to write Upres Layer input height.");
            return false;
          }

        try
          {
            fp.writeInt(n);                                         //  (int) Write number of layer pools from file
          }
        catch(IOException ioErr)
          {
            System.out.println("ERROR: Unable to write number of Upres Layer parameter tuples.");
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
                System.out.println("ERROR: Unable to write Upres Layer name to file.");
                return false;
              }
          }
        buffer = null;                                              //  Release the array

        for(ctr = 0; ctr < n; ctr++)
          {
            try
              {
                fp.writeInt(params[ctr].stride_h);
              }
            catch(IOException ioErr)
              {
                System.out.println("ERROR: Unable to write Upres Layer tuple horizontal stride to file.");
                return false;
              }
            try
              {
                fp.writeInt(params[ctr].stride_v);
              }
            catch(IOException ioErr)
              {
                System.out.println("ERROR: Unable to write Upres Layer tuple vertical stride to file.");
                return false;
              }
            try
              {
                fp.writeInt(params[ctr].padding_h);
              }
            catch(IOException ioErr)
              {
                System.out.println("ERROR: Unable to write Upres Layer tuple horizontal padding to file.");
                return false;
              }
            try
              {
                fp.writeInt(params[ctr].padding_h);
              }
            catch(IOException ioErr)
              {
                System.out.println("ERROR: Unable to write Upres Layer tuple vertical padding to file.");
                return false;
              }
            try
              {
                fp.writeByte((byte)params[ctr].sMethod);
              }
            catch(IOException ioErr)
              {
                System.out.println("ERROR: Unable to write Upres Layer tuple stride method flag to file.");
                return false;
              }
            try
              {
                fp.writeByte((byte)params[ctr].pMethod);
              }
            catch(IOException ioErr)
              {
                System.out.println("ERROR: Unable to write Upres Layer tuple padding method flag to file.");
                return false;
              }
          }

        System.gc();

        return true;
      }

    public class UpresParams
      {
        public int stride_h;                                        //  Horizontal Stride: number of columns to put between input columns
        public int stride_v;                                        //  Vertical Stride: number of rows to put between input rows
        public int padding_h;                                       //  Horizontal Padding: depth of pixels appended to the source border, left and right
        public int padding_v;                                       //  Vertical Padding: depth of pixels appended to the source border, top and bottom

        public int sMethod;                                         //  In {FILL_ZERO, FILL_SAME, FILL_INTERP}
        public int pMethod;                                         //  (Stored as unsigned chars)

        public UpresParams(int stride, int padding)
          {
            stride_h = stride;                                      //  Default to equal horizontal and vertical stride
            stride_v = stride;

            padding_h = padding;                                    //  Default to equal horizontal and vertical padding
            padding_v = padding;

            sMethod = FILL_ZERO;                                    //  Default to filling the strided rows and columns with zero
            pMethod = FILL_ZERO;                                    //  Default to padding the input rows and columns with zero
          }
      }
  }
