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

public class NormalLayer
  {
    private int i;                                                  //  Length of the input vector
    private double m;                                               //  Mu: the mean learned during training
    private double s;                                               //  Sigma: the standard deviation learned during training
    private double g;                                               //  The factor learned during training
    private double b;                                               //  The constant learned during training

    private double out[];
    private String name;

    public NormalLayer(int inputs)
      {
        int ctr;

        i = inputs;
        m = 0.0;                                                    //  Initialize to no effect:
        s = 1.0;                                                    //  y = g * ((x - m) / s) + b
        g = 1.0;
        b = 0.0;
        out = new double[i];
        for(ctr = 0; ctr < i; ctr++)                                //  Blank out buffer
          out[ctr] = 0.0;
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
        name = nameStr;
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

    public int run(double[] x)
      {
        int j;
        for(j = 0; j < i; j++)
          out[j] = g * ((x[j] - m) / s) + b;
        return i;
      }
  }