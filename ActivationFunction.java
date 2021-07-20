/**************************************************************************************************
 Neural Network library, by Eric C. Joyce
***************************************************************************************************/

import java.lang.Math;
import org.jblas.*;
import static org.jblas.DoubleMatrix.*;

public class ActivationFunction
  {
    public static final int RELU                = 0;                //  [ 0.0, inf)
    public static final int LEAKY_RELU          = 1;                //  (-inf, inf)
    public static final int SIGMOID             = 2;                //  ( 0.0, 1.0)
    public static final int HYPERBOLIC_TANGENT  = 3;                //  [-1.0, 1.0]
    public static final int SOFTMAX             = 4;                //  [ 0.0, 1.0]
    public static final int SYMMETRICAL_SIGMOID = 5;                //  (-1.0, 1.0)
    public static final int THRESHOLD           = 6;                //  { 0.0, 1.0}
    public static final int LINEAR              = 7;                //  (-inf, inf)

    /* Apply to a vector */
    public static void relu(DoubleMatrix y)
      {
        int i;
        for(i = 0; i < y.length; i++)
          {
            if(y.get(i) <= 0.0)
              y.put(i, 0.0);
          }
        return;
      }

    /* Apply to a single real */
    public static double relu(double x)
      {
        return (x > 0.0) ? x : 0.0;
      }

    /* Apply to a vector */
    public static void leaky_relu(DoubleMatrix y, double[] alpha)
      {
        int i;
        for(i = 0; i < y.length; i++)
          {
            if(y.get(i) <= 0.0)
              y.put(i, y.get(i) * alpha[i]);
          }
        return;
      }

    /* Apply to a single real */
    public static double leaky_relu(double x, double alpha)
      {
        return (x > 0.0) ? x : x * alpha;
      }

    /* Default to alpha = 1.0 */
    public static double leaky_relu(double x)
      {
        return leaky_relu(x, 1.0);
      }

    /* Apply to a vector */
    public static void sigmoid(DoubleMatrix y, double[] alpha)
      {
        int i;
        for(i = 0; i < y.length; i++)
          y.put(i, 1.0 / (1.0 + Math.pow(Math.E, y.get(i) * alpha[i])));
        return;
      }

    /* Apply to a vector assuming alpha = 1.0-vector */
    public static void sigmoid(DoubleMatrix y)
      {
        int i;
        for(i = 0; i < y.length; i++)
          y.put(i, 1.0 / (1.0 + Math.pow(Math.E, y.get(i))));
        return;
      }

    /* Apply to a single real */
    public static double sigmoid(double x, double alpha)
      {
        return 1.0 / (1.0 + Math.pow(Math.E, -x * alpha));
      }

    /* Default to alpha = 1.0 */
    public static double sigmoid(double x)
      {
        return sigmoid(x, 1.0);
      }

    /* Apply to a vector */
    public static void tanh(DoubleMatrix y, double[] alpha)
      {
        int i;
        for(i = 0; i < y.length; i++)
          y.put(i, (2.0 / (1.0 + Math.pow(Math.E, -2.0 * y.get(i) * alpha[i]))) - 1.0);
        return;
      }

    /* Apply to a single real */
    public static double tanh(double x, double alpha)
      {
        return (2.0 / (1.0 + Math.pow(Math.E, -2.0 * x * alpha))) - 1.0;
      }

    /* Default to alpha = 1.0 */
    public static double tanh(double x)
      {
        return tanh(x, 1.0);
      }

    /* Apply to a vector */
    public static void softmax(DoubleMatrix y)
      {
        int i;
        double softmaxdenom = 0.0;

        for(i = 0; i < y.length; i++)
          softmaxdenom += Math.pow(Math.E, y.get(i));

        for(i = 0; i < y.length; i++)
          y.put(i, Math.pow(Math.E, y.get(i)) / softmaxdenom);

        return;
      }

    /* Apply to a single real */
    public static double softmax(double x, double denom)
      {
        return Math.pow(Math.E, x) / denom;
      }

    /* Apply to a vector */
    public static void sym_sigmoid(DoubleMatrix y, double[] alpha)
      {
        int i;
        for(i = 0; i < y.length; i++)
          y.put(i, (1.0 - Math.pow(Math.E, -y.get(i) * alpha[i])) / (1.0 + Math.pow(Math.E, -y.get(i) * alpha[i])));
        return;
      }

    /* Apply to a single real */
    public static double sym_sigmoid(double x, double alpha)
      {
        return (1.0 - Math.pow(Math.E, -x * alpha)) / (1.0 + Math.pow(Math.E, -x * alpha));
      }

    /* Default to alpha = 1.0 */
    public static double sym_sigmoid(double x)
      {
        return sym_sigmoid(x, 1.0);
      }

    /* Apply to a vector */
    public static void threshold(DoubleMatrix y, double[] alpha)
      {
        int i;
        for(i = 0; i < y.length; i++)
          {
            if(y.get(i) > alpha[i])
              y.put(i, 1.0);
            else
              y.put(i, 0.0);
          }
        return;
      }

    /* Apply to a single real */
    public static double threshold(double x, double alpha)
      {
        return (x > alpha) ? 1.0 : 0.0;
      }

    /* Default to alpha = 1.0 */
    public static double threshold(double x)
      {
        return threshold(x, 1.0);
      }

    /* Apply to a vector */
    public static void linear(DoubleMatrix y, double[] alpha)
      {
        int i;
        for(i = 0; i < y.length; i++)
          y.put(i, y.get(i) * alpha[i]);
        return;
      }

    /* Apply to a single real */
    public static double linear(double x, double alpha)
      {
        return x * alpha;
      }

    /* Default to alpha = 1.0 */
    public static double linear(double x)
      {
        return linear(x, 1.0);
      }
  }