/**************************************************************************************************
 Neural Network library, by Eric C. Joyce

 Accumulation Layers are collection tools, pass-throughs

 Note that this file does NOT seed the randomizer. That should be done by the parent program.
***************************************************************************************************/

public class AccumLayer
  {
    private int i;                                                  //  Number of inputs--ACCUMULATORS GET NO bias-1
    private String name;
    private double out[];

    /* Accumulator Layers have as many outputs as inputs. Additionally set the name. */
    public AccumLayer(int inputs, String nameStr)
      {
        i = inputs;
        out = new double[i];
        name = nameStr;
      }

    /* Accumulator Layers have as many outputs as inputs. */
    public AccumLayer(int inputs)
      {
        this(inputs, "");
      }

    /* Set the name of the Accumulator Layer */
    public void setName(String nameStr)
      {
        name = nameStr;
        return;
      }

    /* Return the layer's output length */
    public int outputLen()
      {
        return i;
      }
  }
