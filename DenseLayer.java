/**************************************************************************************************
 Neural Network library, by Eric C. Joyce

 Model a Dense Layer as two matrices and two vectors:

    input vec{x}         weights W             masks M
 [ x1 x2 x3 x4 1 ]  [ w11 w12 w13 w14 ]  [ m11 m12 m13 m14 ]
                    [ w21 w22 w23 w24 ]  [ m21 m22 m23 m24 ]
                    [ w31 w32 w33 w34 ]  [ m31 m32 m33 m34 ]
                    [ w41 w42 w43 w44 ]  [ m41 m42 m43 m44 ]
                    [ w51 w52 w53 w54 ]  [  1   1   1   1  ]

                    activation function
                         vector f
               [ func1 func2 func3 func4 ]

                     auxiliary vector
                          alpha
               [ param1 param2 param3 param4 ]

 Broadcast W and M = W'
 vec{x} dot W' = x'
 vec{output} is func[i](x'[i], param[i]) for each i

 Not all activation functions need a parameter. It's just a nice feature we like to offer.

 Note that this file does NOT seed the randomizer. That should be done by the parent program.
***************************************************************************************************/

import org.jblas.*;
import static org.jblas.DoubleMatrix.*;

public class DenseLayer
  {
    private int i;                                                  //  Number of inputs--NOT COUNTING the added bias-1
    private int n;                                                  //  Number of processing units in this layer
    private DoubleMatrix W;                                         //  ((i + 1) x n) matrix
    private DoubleMatrix M;                                         //  ((i + 1) x n) matrix, all either 0.0 or 1.0
    private int f[];                                                //  n-array
    private double alpha[];                                         //  n-array
    private String name;
    private DoubleMatrix out;                                       //  n-array

    /*  DenseLayer 'nameStr' shall have 'inputs' inputs and 'nodes' nodes. */
    public DenseLayer(int inputs, int nodes, String nameStr)
      {
        int x, y;

        i = inputs;                                                 //  Set this layer's inputs
        n = nodes;                                                  //  Set this layer's number of nodes
        W = new DoubleMatrix(i + 1, n);                             //  Allocate this layer's weight matrix
        M = new DoubleMatrix(i + 1, n);                             //  Allocate this layer's mask matrix
        f = new int[n];                                             //  Allocate this layer's function-flag array
        alpha = new double[n];                                      //  Allocate this layer's function-parameter array
        out = new DoubleMatrix(n);                                  //  Allocate output buffer

        for(y = 0; y < (i + 1); y++)
          {
            for(x = 0; x < n; x++)
              {
                W.put(y, x, -1.0 + Math.random() * 2.0);            //  Generate random numbers in [ -1.0, 1.0 ]
                M.put(y, x, 1.0);                                   //  All are UNmasked
              }
          }

        for(x = 0; x < n; x++)                                      //  Default all to ReLU with parameter = 1.0
          {
            f[x] = ActivationFunction.RELU;
            alpha[x] = 1.0;
          }

        for(x = 0; x < n; x++)                                      //  Blank out 'out' array
          out.put(x, 0.0);

        name = nameStr;                                             //  Blank out layer name
      }

    /*  New DenseLayer shall have 'inputs' inputs and 'nodes' nodes. */
    public DenseLayer(int inputs, int nodes)
      {
        this(inputs, nodes, "");
      }

    /* Set entirety of layer's weight matrix.
       Input buffer 'w' is expected to be ROW-MAJOR even though the internal W is column-major
            weights W
       [ w0  w1  w2  w3  ]
       [ w4  w5  w6  w7  ]
       [ w8  w9  w10 w11 ]
       [ w12 w13 w14 w15 ]
       [ w16 w17 w18 w19 ]  <--- biases  */
    public void setW(double[] w)
      {
        int x, y;
        for(y = 0; y < (i + 1); y++)
          {
            for(x = 0; x < n; x++)
              W.put(y, x, w[y * n + x]);
          }
        return;
      }

    /* Set entirety of weights for i-th column/neuron/unit. */
    public void setW_i(double[] w, int index)
      {
        int ctr;
        for(ctr = 0; ctr <= i; ctr++)
          W.put(ctr, index, w[ctr]);
        return;
      }

    /*  Set element [i, j] of layer's weight matrix */
    public void setW_ij(double w, int index_i, int index_j)
      {
        if(index_j * n + index_i < (i + 1) * n)
          W.put(index_i, index_j, w);
        return;
      }

    /*  Set entirety of layer's mask matrix */
    public void setM(boolean[] m)
      {
        int x, y;
        for(y = 0; y < (i + 1); y++)
          {
            for(x = 0; x < n; x++)
              {
                if(m[y * n + x])
                  M.put(y, x, 1.0);
                else
                  M.put(y, x, 0.0);
              }
          }
        return;
      }

    /*  Set entirety of masks for i-th column/neuron/unit */
    public void setM_i(boolean[] m, int index)
      {
        int ctr;
        for(ctr = 0; ctr <= i; ctr++)
          {
            if(m[ctr])
              M.put(ctr, index, 1.0);
            else
              M.put(ctr, index, 0.0);
          }
        return;
      }

    /*  Set element [i, j] of layer's mask matrix */
    public void setM_ij(boolean m, int index_i, int index_j)
      {
        if(index_j * n + index_i < (i + 1) * n)
          {
            if(m)
              M.put(index_i, index_j, 1.0);
            else
              M.put(index_i, index_j, 0.0);
          }
        return;
      }

    /*  Set activation function of i-th neuron/unit */
    public void setF_i(int func, int index)
      {
        if(index < n)
          f[index] = func;
        return;
      }

    /*  Set activation function auxiliary parameter of i-th neuron/unit */
    public void setA_i(double a, int index)
      {
        if(index < n)
          alpha[index] = a;
        return;
      }

    public void setName(String nameStr)
      {
        name = nameStr;
        return;
      }

    public void print()
      {
        int x, y;

        for(x = 0; x < i + 1; x++)
          {
            if(x == i)
              System.out.print("bias [");
            else
              System.out.print("     [");
            for(y = 0; y < n; y++)
              {
                if(W.get(y, x) >= 0.0)
                  System.out.printf(" %.5f ", W.get(y, x));
                else
                  System.out.printf("%.5f ", W.get(y, x));
              }
            System.out.print("]\n");
          }
        System.out.printf("f = [");
        for(x = 0; x < n; x++)
          {
            switch(f[x])
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
          }
        System.out.print("]\n");
        System.out.print("a = [");
        for(x = 0; x < n; x++)
          System.out.printf("%.4f ", alpha[x]);
        System.out.print("]\n");

        return;
      }

    /* Return the layer's output length
       (For Dense layers, this is the number of units) */
    public int outputLen()
      {
        return n;
      }

    /*  Run the given input vector 'x' of length 'layer'->'i' through the DenseLayer 'layer'.
        Output is stored internally in layer->out. */
    public int run(double[] xvec)
      {
                                                                    //  Input vector augmented with additional (bias) 1.0
        DoubleMatrix xprime = new DoubleMatrix(i + 1);              //  (1 * (length-of-input + 1))
        DoubleMatrix Wprime = new DoubleMatrix(i + 1, n);           //  ((length-of-input + 1) * nodes)
        int x, y;

        for(x = 0; x < i; x++)                                      //  Append 1.0 to input vector
          xprime.put(x, xvec[x]);
        xprime.put(x, 1.0);

        //                       weights W                                                  masks M
        //     i = 0 ----------------------------> layer->nodes        i = 0 ----------------------------> layer->nodes
        //   j   [ A+0      A+((len+1)*i)        A+((len+1)*i)+j ]   j   [ A+0      A+((len+1)*i)        A+((len+1)*i)+j ]
        //   |   [ A+1      A+((len+1)*i)+1      A+((len+1)*i)+j ]   |   [ A+1      A+((len+1)*i)+1      A+((len+1)*i)+j ]
        //   |   [ A+2      A+((len+1)*i)+2      A+((len+1)*i)+j ]   |   [ A+2      A+((len+1)*i)+2      A+((len+1)*i)+j ]
        //   |   [ ...      ...                  ...             ]   |   [ ...      ...                  ...             ]
        //   V   [ A+len    A+((len+1)*i)+len    A+((len+1)*i)+j ]   V   [ A+len    A+((len+1)*i)+len    A+((len+1)*i)+j ]
        // len+1 [ A+len+1  A+((len+1)*i)+len+1  A+((len+1)*i)+j ] len+1 [  1        1                    1              ]
        Wprime = W.mul(M);                                          //  Broadcast weights and masks into W'

        out = Wprime.mmul(xprime);                                  //  out = Wprime dot xprime

        for(x = 0; x < n; x++)                                      //  Run each element in out through appropriate function
          {                                                         //  with corresponding parameter
            switch(f[x])
              {
                case ActivationFunction.RELU:                ActivationFunction.relu(out);  break;
                case ActivationFunction.LEAKY_RELU:          ActivationFunction.leaky_relu(out, alpha);  break;
                case ActivationFunction.SIGMOID:             ActivationFunction.sigmoid(out, alpha);  break;
                case ActivationFunction.HYPERBOLIC_TANGENT:  ActivationFunction.tanh(out, alpha);  break;
                case ActivationFunction.SOFTMAX:             ActivationFunction.softmax(out);  break;
                case ActivationFunction.SYMMETRICAL_SIGMOID: ActivationFunction.sym_sigmoid(out, alpha);  break;
                case ActivationFunction.THRESHOLD:           ActivationFunction.threshold(out, alpha);  break;
                default:                                     ActivationFunction.linear(out, alpha);
              }
          }

        xprime = null;                                              //  Force release of allocated memory
        Wprime = null;
        System.gc();                                                //  Call the garbage collector

        return n;                                                   //  Return the length of layer->out
      }
  }
