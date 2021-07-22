/**************************************************************************************************
 Neural Network library, by Eric C. Joyce

 Note that this file does NOT seed the randomizer. That should be done by the parent program.
***************************************************************************************************/

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;

public class NeuralNet
  {
    public static final int INPUT_ARRAY  = 0;                       //  Flag refers to network input
    public static final int DENSE_ARRAY  = 1;                       //  Flag refers to 'denselayers'
    public static final int CONV2D_ARRAY = 2;                       //  Flag refers to 'convlayers'
    public static final int ACCUM_ARRAY  = 3;                       //  Flag refers to 'accumlayers'
    public static final int LSTM_ARRAY   = 4;                       //  Flag refers to 'lstmlayers'
    public static final int GRU_ARRAY    = 5;                       //  Flag refers to 'grulayers'
    public static final int POOL_ARRAY   = 6;                       //  Flag refers to 'upreslayers'
    public static final int UPRES_ARRAY  = 7;                       //  Flag refers to 'poollayers'
    public static final int NORMAL_ARRAY = 8;                       //  Flag refers to 'normlayers'

    public static final int VARSTR_LEN     = 16;                    //  Length of a Variable key string
    public static final int LAYER_NAME_LEN = 32;                    //  Length of a Layer 'name' string
    public static final int COMMSTR_LEN    = 64;                    //  Length of a Network Comment string

    private int inputs;                                             //  Number of Network inputs

    private Edge edgelist[];                                        //  Edge list
    private int len;                                                //  Length of that array
    private DenseLayer denselayers[];                               //  Array of Dense Layers
    private int denseLen;                                           //  Length of that array
    private Conv2DLayer convlayers[];                               //  Array of Conv2D Layers
    private int convLen;                                            //  Length of that array
    private AccumLayer accumlayers[];                               //  Array of Accum Layers
    private int accumLen;                                           //  Length of that array
    private LSTMLayer lstmlayers[];                                 //  Array of LSTM Layers
    private int lstmLen;                                            //  Length of that array
    private GRULayer grulayers[];                                   //  Array of GRU Layers
    private int gruLen;                                             //  Length of that array
    private PoolLayer poollayers[];                                 //  Array of Pooling Layers
    private int poolLen;                                            //  Length of that array
    private UpresLayer upreslayers[];                               //  Array of Up-res Layers
    private int upresLen;                                           //  Length of that array
    private NormalLayer normlayers[];                               //  Array of Normalization Layers
    private int normalLen;                                          //  Length of that array

    private Variable variables[];                                   //  Array of Network Variables
    private int vars;                                               //  Length of that array

    private int gen;                                                //  Network generation/epoch
    private double fit;                                             //  Network fitness
    private String comment;                                         //  Network comment

    /* Initialize a deep network for ad hoc construction. */
    public NeuralNet(int num_inputs)
      {
        if(num_inputs > 0)
          inputs = num_inputs;                                      //  Save number of inputs
        else
          inputs = 1;

        len = 0;                                                    //  Initially, no edges
        denseLen = 0;                                               //  Initially, no Dense Layers
        convLen = 0;                                                //  Initially, no Convolutional Layers
        accumLen = 0;                                               //  Initially, no Accumulation Layers
        lstmLen = 0;                                                //  Initially, no LSTM Layers
        gruLen = 0;                                                 //  Initially, no GRU Layers
        poolLen = 0;                                                //  Initially, no Pooling Layers
        upresLen = 0;                                               //  Initially, no Up-resolution Layers
        normalLen = 0;                                              //  Initially, no Normalization Layers

        vars = 0;                                                   //  Initially, no Variables
        gen = 0;                                                    //  Initialize generation to zero
        fit = 0.0;                                                  //  Initialize fitness to zero
        comment = "";
      }

    /* Input vector 'x' has length = 'inputs'.
       Output vector 'z' will have length = # of units/outputs in last layer */
    public double[] run(double[] x)
      {
        double in[];
        double z[];

        int inLen = 0;
        int outLen = 0;
        int i, j, k, l;
        int last = 0;
        int t;                                                      //  For reading from LSTMs

        i = 0;
        while(i < len)                                              //  For each edge in edgelist
          {
                                                                    //  Set the length of the input vector to the input
                                                                    //  size of the current destination layer.
            switch(edgelist[i].dstType)                             //  Which array contains the destination layer?
              {
                case DENSE_ARRAY:   inLen = denselayers[edgelist[i].dstIndex].inputs();
                                    break;
                case CONV2D_ARRAY:  inLen = convlayers[edgelist[i].dstIndex].width() *
                                            convlayers[edgelist[i].dstIndex].height();
                                    break;
                case ACCUM_ARRAY:   inLen = accumlayers[edgelist[i].dstIndex].inputs();
                                    break;
                case LSTM_ARRAY:    inLen = lstmlayers[edgelist[i].dstIndex].inputDimensionality();
                                    break;
                case GRU_ARRAY:     inLen = grulayers[edgelist[i].dstIndex].inputDimensionality();
                                    break;
                case POOL_ARRAY:    inLen = poollayers[edgelist[i].dstIndex].width() *
                                            poollayers[edgelist[i].dstIndex].height();
                                    break;
                case UPRES_ARRAY:   inLen = upreslayers[edgelist[i].dstIndex].width() *
                                            upreslayers[edgelist[i].dstIndex].height();
                                    break;
                case NORMAL_ARRAY:  inLen = normlayers[edgelist[i].dstIndex].inputs();
                                    break;
              }

            in = new double[inLen];                                 //  Allocate the vector

            k = 0;                                                  //  Point to head of input buffer
            j = i;                                                  //  Advance j to encompass all inputs to current layer
            while(j < len && edgelist[i].dstType == edgelist[j].dstType && edgelist[i].dstIndex == edgelist[j].dstIndex)
              {
                switch(edgelist[j].srcType)
                  {
                    case INPUT_ARRAY:                               //  Receiving from network input
                                       for(l = edgelist[j].selectorStart; l < edgelist[j].selectorEnd; l++)
                                         {
                                           in[k] = x[l];
                                           k++;
                                         }
                                       break;
                    case DENSE_ARRAY:                               //  Receiving from a dense layer
                                       z = denselayers[edgelist[j].srcIndex].output();
                                       for(l = edgelist[j].selectorStart; l < edgelist[j].selectorEnd; l++)
                                         {
                                           in[k] = z[l];
                                           k++;
                                         }
                                       break;
                    case CONV2D_ARRAY:                              //  Receiving from a convolutional layer
                                       z = convlayers[edgelist[j].srcIndex].output();
                                       for(l = edgelist[j].selectorStart; l < edgelist[j].selectorEnd; l++)
                                         {
                                           in[k] = z[l];
                                           k++;
                                         }
                                       break;
                    case ACCUM_ARRAY:                               //  Receiving from an accumulator layer
                                       z = accumlayers[edgelist[j].srcIndex].output();
                                       for(l = edgelist[j].selectorStart; l < edgelist[j].selectorEnd; l++)
                                         {
                                           in[k] = z[l];
                                           k++;
                                         }
                                       break;
                    case LSTM_ARRAY:                                //  Receiving from an LSTM layer
                                       z = lstmlayers[edgelist[j].srcIndex].state();
                                       if(lstmlayers[edgelist[j].srcIndex].timestep() >= lstmlayers[edgelist[j].srcIndex].cacheLen())
                                         t = lstmlayers[edgelist[j].srcIndex].cacheLen() - 1;
                                       else
                                         t = lstmlayers[edgelist[j].srcIndex].timestep() - 1;
                                       for(l = edgelist[j].selectorStart; l < edgelist[j].selectorEnd; l++)
                                         {
                                           in[k] = z[l];            //  Read from the LAST time step
                                           k++;
                                         }
                                       break;
                    case GRU_ARRAY:                                 //  Receiving from a GRU layer
                                       z = grulayers[edgelist[j].srcIndex].state();
                                       if(grulayers[edgelist[j].srcIndex].timestep() >= grulayers[edgelist[j].srcIndex].cacheLen())
                                         t = grulayers[edgelist[j].srcIndex].cacheLen() - 1;
                                       else
                                         t = grulayers[edgelist[j].srcIndex].timestep() - 1;
                                       for(l = edgelist[j].selectorStart; l < edgelist[j].selectorEnd; l++)
                                         {
                                           in[k] = z[l];            //  Read from the LAST time step
                                           k++;
                                         }
                                       break;
                    case POOL_ARRAY:                                //  Receiving from a pooling layer
                                       z = poollayers[edgelist[j].srcIndex].output();
                                       for(l = edgelist[j].selectorStart; l < edgelist[j].selectorEnd; l++)
                                         {
                                           in[k] = z[l];
                                           k++;
                                         }
                                       break;
                    case UPRES_ARRAY:                               //  Receiving from an upres layer
                                       z = upreslayers[edgelist[j].srcIndex].output();
                                       for(l = edgelist[j].selectorStart; l < edgelist[j].selectorEnd; l++)
                                         {
                                           in[k] = z[l];
                                           k++;
                                         }
                                       break;
                    case NORMAL_ARRAY:                              //  Receiving from a normalization layer
                                       z = normlayers[edgelist[j].srcIndex].output();
                                       for(l = edgelist[j].selectorStart; l < edgelist[j].selectorEnd; l++)
                                         {
                                           in[k] = z[l];
                                           k++;
                                         }
                                       break;
                  }
                j++;
              }

            switch(edgelist[i].dstType)                             //  Which array contains the destination layer?
              {
                case DENSE_ARRAY:   outLen = denselayers[edgelist[i].dstIndex].run(in);
                                    break;
                case CONV2D_ARRAY:  outLen = convlayers[edgelist[i].dstIndex].run(in);
                                    break;
                case ACCUM_ARRAY:   outLen = inLen;
                                    for(k = 0; k < inLen; k++)
                                      accumlayers[edgelist[i].dstIndex].set(k, in[k]);
                                    break;
                case LSTM_ARRAY:    outLen = lstmlayers[edgelist[i].dstIndex].run(in);
                                    break;
                case GRU_ARRAY:     outLen = grulayers[edgelist[i].dstIndex].run(in);
                                    break;
                case POOL_ARRAY:    outLen = poollayers[edgelist[i].dstIndex].run(in);
                                    break;
                case UPRES_ARRAY:   outLen = upreslayers[edgelist[i].dstIndex].run(in);
                                    break;
                case NORMAL_ARRAY:  outLen = normlayers[edgelist[i].dstIndex].run(in);
                                    break;
              }

            in = null;                                              //  Release input vector

            last = i;                                               //  Save the index of the previous edge
            i = j;                                                  //  Increment 'i'
          }

        z = new double[outLen];                                     //  Copy from last (internal) out to 'z'

        switch(edgelist[last].dstType)
          {
            case DENSE_ARRAY:   z = denselayers[edgelist[last].dstIndex].output();
                                break;
            case CONV2D_ARRAY:  z = convlayers[edgelist[last].dstIndex].output();
                                break;
            case ACCUM_ARRAY:   z = accumlayers[edgelist[last].dstIndex].output();
                                break;
            case LSTM_ARRAY:    if(lstmlayers[edgelist[last].dstIndex].timestep() >= lstmlayers[edgelist[last].dstIndex].cacheLen())
                                  t = lstmlayers[edgelist[last].dstIndex].cacheLen() - 1;
                                else
                                  t = lstmlayers[edgelist[last].dstIndex].timestep() - 1;
                                z = lstmlayers[edgelist[last].dstIndex].state(t);
                                break;
            case GRU_ARRAY:     if(grulayers[edgelist[last].dstIndex].timestep() >= grulayers[edgelist[last].dstIndex].cacheLen())
                                  t = grulayers[edgelist[last].dstIndex].cacheLen() - 1;
                                else
                                  t = grulayers[edgelist[last].dstIndex].timestep() - 1;
                                z = grulayers[edgelist[last].dstIndex].state(t);
                                break;
            case POOL_ARRAY:    z = poollayers[edgelist[last].dstIndex].output();
                                break;
            case UPRES_ARRAY:   z = upreslayers[edgelist[last].dstIndex].output();
                                break;
            case NORMAL_ARRAY:  z = normlayers[edgelist[last].dstIndex].output();
                                break;
          }

        return z;
      }

    /* Connect layer srcIndex to layer dstIndex.
       We have to identify the types of src and dst so that we can identify which arrays the layer structs are in.
       We specify the slice from src as [selectorStart, selectorEnd].
       e.g.  linkLayers(INPUT_ARRAY, 0, 0, 63, CONV2D_ARRAY, 0)     //  From input[0:63] to convolution layer
             linkLayers(CONV2D_ARRAY, 0, 0, 91, ACCUM_ARRAY, 0)     //  From convolution layer to accumulator
             linkLayers(INPUT_ARRAY, 0, 63, 64, ACCUM_ARRAY, 0)     //  From input[63:64] to accumulator
             linkLayers(ACCUM_ARRAY, 0, 0, 92, DENSE_ARRAY, 0)      //  From accumulator to dense layer
             linkLayers(DENSE_ARRAY, 0, 0, 40, DENSE_ARRAY, 1)      //  From dense layer to dense layer
             linkLayers(DENSE_ARRAY, 1, 0, 10, DENSE_ARRAY, 2)      //  From dense layer to dense layer */
    public boolean linkLayers(int srcFlag, int src, int selectorStart, int selectorEnd, int dstFlag, int dst)
      {
        int i, j;
        ArrayList<Node> queue;
        ArrayList<Node> tmp;
        Node node;
        ArrayList<Node> visited;
        int vlen = 0;
        Edge tmp_edgelist[];

        if(srcFlag == DENSE_ARRAY && src >= denseLen)               //  Is the request out of bounds?
          return false;
        if(srcFlag == CONV2D_ARRAY && src >= convLen)
          return false;
        if(srcFlag == ACCUM_ARRAY && src >= accumLen)
          return false;
        if(srcFlag == LSTM_ARRAY && src >= lstmLen)
          return false;
        if(srcFlag == GRU_ARRAY && src >= gruLen)
          return false;
        if(srcFlag == POOL_ARRAY && src >= poolLen)
          return false;
        if(srcFlag == UPRES_ARRAY && src >= upresLen)
          return false;
        if(srcFlag == NORMAL_ARRAY && src >= normalLen)
          return false;

        if(dstFlag == DENSE_ARRAY && dst >= denseLen)
          return false;
        if(dstFlag == CONV2D_ARRAY && dst >= convLen)
          return false;
        if(dstFlag == ACCUM_ARRAY && dst >= accumLen)
          return false;
        if(dstFlag == LSTM_ARRAY && dst >= lstmLen)
          return false;
        if(dstFlag == GRU_ARRAY && dst >= gruLen)
          return false;
        if(dstFlag == POOL_ARRAY && dst >= poolLen)
          return false;
        if(dstFlag == UPRES_ARRAY && dst >= upresLen)
          return false;
        if(dstFlag == NORMAL_ARRAY && dst >= normalLen)
          return false;

        if(srcFlag == DENSE_ARRAY && selectorStart >= denselayers[src].nodes())
          return false;
        if(srcFlag == CONV2D_ARRAY && selectorStart >= convlayers[src].outputLen())
          return false;
        if(srcFlag == ACCUM_ARRAY && selectorStart >= accumlayers[src].inputs())
          return false;
        if(srcFlag == LSTM_ARRAY && selectorStart >= lstmlayers[src].stateDimensionality())
          return false;
        if(srcFlag == GRU_ARRAY && selectorStart >= grulayers[src].stateDimensionality())
          return false;
        if(srcFlag == POOL_ARRAY && selectorStart >= poollayers[src].outputLen())
          return false;
        if(srcFlag == UPRES_ARRAY && selectorStart >= upreslayers[src].outputLen())
          return false;
        if(srcFlag == NORMAL_ARRAY && selectorStart >= normlayers[src].inputs())
          return false;

        if(srcFlag == DENSE_ARRAY && selectorEnd > denselayers[src].nodes())
          return false;
        if(srcFlag == CONV2D_ARRAY && selectorEnd > convlayers[src].outputLen())
          return false;
        if(srcFlag == ACCUM_ARRAY && selectorEnd > accumlayers[src].outputLen())
          return false;
        if(srcFlag == LSTM_ARRAY && selectorEnd > lstmlayers[src].stateDimensionality())
          return false;
        if(srcFlag == GRU_ARRAY && selectorEnd > grulayers[src].stateDimensionality())
          return false;
        if(srcFlag == POOL_ARRAY && selectorEnd > poollayers[src].outputLen())
          return false;
        if(srcFlag == UPRES_ARRAY && selectorEnd > upreslayers[src].outputLen())
          return false;
        if(srcFlag == NORMAL_ARRAY && selectorEnd > normlayers[src].inputs())
          return false;                                             //  A NormalLayer's number of outputs = its number of inputs

        if(selectorEnd < selectorStart)
          return false;
                                                                    //  Check output-input shapes match
        if(srcFlag == DENSE_ARRAY && dstFlag == DENSE_ARRAY &&      //  Dense-->Dense
           denselayers[src].outputLen() != denselayers[dst].inputs())
          return false;
        if(srcFlag == DENSE_ARRAY && dstFlag == CONV2D_ARRAY &&     //  Dense-->Conv2D
           denselayers[src].outputLen() != convlayers[dst].width() * convlayers[dst].height())
          return false;
        if(srcFlag == DENSE_ARRAY && dstFlag == ACCUM_ARRAY &&      //  Dense-->Accumulator
           denselayers[src].outputLen() > accumlayers[dst].inputs())
          return false;
        if(srcFlag == DENSE_ARRAY && dstFlag == LSTM_ARRAY &&       //  Dense-->LSTM
           denselayers[src].outputLen() > lstmlayers[dst].inputDimensionality())
          return false;
        if(srcFlag == DENSE_ARRAY && dstFlag == GRU_ARRAY &&        //  Dense-->GRU
           denselayers[src].outputLen() > grulayers[dst].inputDimensionality())
          return false;
        if(srcFlag == DENSE_ARRAY && dstFlag == POOL_ARRAY &&       //  Dense-->Pool
           denselayers[src].outputLen() != poollayers[dst].width() * poollayers[dst].height())
          return false;
        if(srcFlag == DENSE_ARRAY && dstFlag == UPRES_ARRAY &&      //  Dense-->Upres
           denselayers[src].outputLen() != upreslayers[dst].width() * upreslayers[dst].height())
          return false;
        if(srcFlag == DENSE_ARRAY && dstFlag == NORMAL_ARRAY &&     //  Dense-->Normalization
           denselayers[src].outputLen() != normlayers[dst].inputs())
          return false;

        if(srcFlag == CONV2D_ARRAY && dstFlag == DENSE_ARRAY &&     //  Conv2D-->Dense
           convlayers[src].outputLen() != denselayers[dst].inputs())
          return false;
        if(srcFlag == CONV2D_ARRAY && dstFlag == CONV2D_ARRAY &&    //  Conv2D-->Conv2D
           convlayers[src].outputLen() != convlayers[dst].width() * convlayers[dst].height())
          return false;
        if(srcFlag == CONV2D_ARRAY && dstFlag == ACCUM_ARRAY &&     //  Conv2D-->Accumulator
           convlayers[src].outputLen() > accumlayers[dst].inputs())
          return false;
        if(srcFlag == CONV2D_ARRAY && dstFlag == LSTM_ARRAY &&      //  Conv2D-->LSTM
           convlayers[src].outputLen() > lstmlayers[dst].inputDimensionality())
          return false;
        if(srcFlag == CONV2D_ARRAY && dstFlag == GRU_ARRAY &&       //  Conv2D-->GRU
           convlayers[src].outputLen() > grulayers[dst].inputDimensionality())
          return false;
        if(srcFlag == CONV2D_ARRAY && dstFlag == POOL_ARRAY &&      //  Conv2D-->Pool
           convlayers[src].outputLen() != poollayers[dst].width() * poollayers[dst].height())
          return false;
        if(srcFlag == CONV2D_ARRAY && dstFlag == UPRES_ARRAY &&     //  Conv2D-->Upres
           convlayers[src].outputLen() != upreslayers[dst].width() * upreslayers[dst].height())
          return false;
        if(srcFlag == CONV2D_ARRAY && dstFlag == NORMAL_ARRAY &&    //  Conv2D-->Normalization
           convlayers[src].outputLen() != normlayers[dst].inputs())
          return false;

        if(srcFlag == ACCUM_ARRAY && dstFlag == DENSE_ARRAY &&      //  Accumulator-->Dense
           accumlayers[src].inputs() != denselayers[dst].inputs())
          return false;
        if(srcFlag == ACCUM_ARRAY && dstFlag == CONV2D_ARRAY &&     //  Accumulator-->Conv2D
           accumlayers[src].inputs() != convlayers[dst].width() * convlayers[dst].height())
          return false;
        if(srcFlag == ACCUM_ARRAY && dstFlag == ACCUM_ARRAY &&      //  Accumulator-->Accumulator
           accumlayers[src].inputs() > accumlayers[dst].inputs())   //  Incoming layer free to be < Accumulator size
          return false;
        if(srcFlag == ACCUM_ARRAY && dstFlag == LSTM_ARRAY &&       //  Accumulator-->LSTM
           accumlayers[src].inputs() != lstmlayers[dst].inputDimensionality())
          return false;
        if(srcFlag == ACCUM_ARRAY && dstFlag == GRU_ARRAY &&        //  Accumulator-->GRU
           accumlayers[src].inputs() != grulayers[dst].inputDimensionality())
          return false;
        if(srcFlag == ACCUM_ARRAY && dstFlag == POOL_ARRAY &&       //  Accumulator-->Pool
           accumlayers[src].inputs() != poollayers[dst].width() * poollayers[dst].height())
          return false;
        if(srcFlag == ACCUM_ARRAY && dstFlag == UPRES_ARRAY &&      //  Accumulator-->Upres
           accumlayers[src].inputs() != upreslayers[dst].width() * upreslayers[dst].height())
          return false;
        if(srcFlag == ACCUM_ARRAY && dstFlag == NORMAL_ARRAY &&     //  Accumulator-->Normalization
           accumlayers[src].inputs() != normlayers[dst].inputs())
          return false;

        if(srcFlag == LSTM_ARRAY && dstFlag == DENSE_ARRAY &&       //  LSTM-->Dense
           lstmlayers[src].stateDimensionality() != denselayers[dst].inputs())
          return false;
        if(srcFlag == LSTM_ARRAY && dstFlag == CONV2D_ARRAY &&      //  LSTM-->Conv2D
           lstmlayers[src].stateDimensionality() != convlayers[dst].width() * convlayers[dst].height())
          return false;
        if(srcFlag == LSTM_ARRAY && dstFlag == ACCUM_ARRAY &&       //  LSTM-->Accumulator : Incoming layer free to be < Accumulator size
           lstmlayers[src].stateDimensionality() > accumlayers[dst].inputs())
          return false;
        if(srcFlag == LSTM_ARRAY && dstFlag == LSTM_ARRAY &&        //  LSTM-->LSTM
           lstmlayers[src].stateDimensionality() != lstmlayers[dst].inputDimensionality())
          return false;
        if(srcFlag == LSTM_ARRAY && dstFlag == GRU_ARRAY &&         //  LSTM-->GRU
           lstmlayers[src].stateDimensionality() != grulayers[dst].inputDimensionality())
          return false;
        if(srcFlag == LSTM_ARRAY && dstFlag == POOL_ARRAY &&        //  LSTM-->Pool
           lstmlayers[src].stateDimensionality() != poollayers[dst].width() * poollayers[dst].height())
          return false;
        if(srcFlag == LSTM_ARRAY && dstFlag == UPRES_ARRAY &&       //  LSTM-->Upres
           lstmlayers[src].stateDimensionality() != upreslayers[dst].width() * upreslayers[dst].height())
          return false;
        if(srcFlag == LSTM_ARRAY && dstFlag == NORMAL_ARRAY &&      //  LSTM-->Normalization
           lstmlayers[src].stateDimensionality() != normlayers[dst].inputs())
          return false;

        if(srcFlag == GRU_ARRAY && dstFlag == DENSE_ARRAY &&        //  GRU-->Dense
           grulayers[src].stateDimensionality() != denselayers[dst].inputs())
          return false;
        if(srcFlag == GRU_ARRAY && dstFlag == CONV2D_ARRAY &&       //  GRU-->Conv2D
           grulayers[src].stateDimensionality() != convlayers[dst].width() * convlayers[dst].height())
          return false;
        if(srcFlag == GRU_ARRAY && dstFlag == ACCUM_ARRAY &&        //  GRU-->Accumulator : Incoming layer free to be < Accumulator size
           grulayers[src].stateDimensionality() > accumlayers[dst].inputs())
          return false;
        if(srcFlag == GRU_ARRAY && dstFlag == LSTM_ARRAY &&         //  GRU-->LSTM
           grulayers[src].stateDimensionality() != lstmlayers[dst].inputDimensionality())
          return false;
        if(srcFlag == GRU_ARRAY && dstFlag == GRU_ARRAY &&          //  GRU-->GRU
           grulayers[src].stateDimensionality() != grulayers[dst].inputDimensionality())
          return false;
        if(srcFlag == GRU_ARRAY && dstFlag == POOL_ARRAY &&         //  GRU-->Pool
           grulayers[src].stateDimensionality() != poollayers[dst].width() * poollayers[dst].height())
          return false;
        if(srcFlag == GRU_ARRAY && dstFlag == UPRES_ARRAY &&        //  GRU-->Upres
           grulayers[src].stateDimensionality() != upreslayers[dst].width() * upreslayers[dst].height())
          return false;
        if(srcFlag == GRU_ARRAY && dstFlag == NORMAL_ARRAY &&       //  GRU-->Normalization
           grulayers[src].stateDimensionality() != normlayers[dst].inputs())
          return false;

        if(srcFlag == POOL_ARRAY && dstFlag == DENSE_ARRAY &&       //  Pool-->Dense
           poollayers[src].outputLen() != denselayers[dst].inputs())
          return false;
        if(srcFlag == POOL_ARRAY && dstFlag == CONV2D_ARRAY &&      //  Pool-->Conv2D
           poollayers[src].outputLen() != convlayers[dst].width() * convlayers[dst].height())
          return false;
        if(srcFlag == POOL_ARRAY && dstFlag == ACCUM_ARRAY &&       //  Pool-->Accumulator
           poollayers[src].outputLen() > accumlayers[dst].inputs()) //  Incoming layer free to be < Accumulator size
          return false;
        if(srcFlag == POOL_ARRAY && dstFlag == LSTM_ARRAY &&        //  Pool-->LSTM
           poollayers[src].outputLen() != lstmlayers[dst].inputDimensionality())
          return false;
        if(srcFlag == POOL_ARRAY && dstFlag == GRU_ARRAY &&         //  Pool-->GRU
           poollayers[src].outputLen() != grulayers[dst].inputDimensionality())
          return false;
        if(srcFlag == POOL_ARRAY && dstFlag == POOL_ARRAY &&        //  Pool-->Pool
           poollayers[src].outputLen() != poollayers[dst].width() * poollayers[dst].height())
          return false;
        if(srcFlag == POOL_ARRAY && dstFlag == UPRES_ARRAY &&       //  Pool-->Upres
           poollayers[src].outputLen() != upreslayers[dst].width() * upreslayers[dst].height())
          return false;
        if(srcFlag == POOL_ARRAY && dstFlag == NORMAL_ARRAY &&      //  Pool-->Normalization
           poollayers[src].outputLen() != normlayers[dst].inputs())
          return false;

        if(srcFlag == UPRES_ARRAY && dstFlag == DENSE_ARRAY &&      //  Upres-->Dense
           upreslayers[src].outputLen() != denselayers[dst].inputs())
          return false;
        if(srcFlag == UPRES_ARRAY && dstFlag == CONV2D_ARRAY &&     //  Upres-->Conv2D
           upreslayers[src].outputLen() != convlayers[dst].width() * convlayers[dst].height())
          return false;
        if(srcFlag == UPRES_ARRAY && dstFlag == ACCUM_ARRAY &&      //  Upres-->Accumulator
           upreslayers[src].outputLen() > accumlayers[dst].inputs())//  Incoming layer free to be < Accumulator size
          return false;
        if(srcFlag == UPRES_ARRAY && dstFlag == LSTM_ARRAY &&       //  Upres-->LSTM
           upreslayers[src].outputLen() != lstmlayers[dst].inputDimensionality())
          return false;
        if(srcFlag == UPRES_ARRAY && dstFlag == GRU_ARRAY &&        //  Upres-->GRU
           upreslayers[src].outputLen() != grulayers[dst].inputDimensionality())
          return false;
        if(srcFlag == UPRES_ARRAY && dstFlag == POOL_ARRAY &&       //  Upres-->Pool
           upreslayers[src].outputLen() != poollayers[dst].width() * poollayers[dst].height())
          return false;
        if(srcFlag == UPRES_ARRAY && dstFlag == UPRES_ARRAY &&      //  Upres-->Upres
           upreslayers[src].outputLen() != upreslayers[dst].width() * upreslayers[dst].height())
          return false;
        if(srcFlag == UPRES_ARRAY && dstFlag == NORMAL_ARRAY &&     //  Upres-->Normalization
           upreslayers[src].outputLen() != normlayers[dst].inputs())
          return false;

        if(srcFlag == NORMAL_ARRAY && dstFlag == DENSE_ARRAY &&     //  Normalization-->Dense
           normlayers[src].inputs() != denselayers[dst].inputs())
          return false;
        if(srcFlag == NORMAL_ARRAY && dstFlag == CONV2D_ARRAY &&    //  Normalization-->Conv2D
           normlayers[src].inputs() != convlayers[dst].width() * convlayers[dst].height())
          return false;
        if(srcFlag == NORMAL_ARRAY && dstFlag == ACCUM_ARRAY &&     //  Normalization-->Accumulator
           normlayers[src].inputs() > accumlayers[dst].inputs())    //  Incoming layer free to be < Accumulator size
          return false;
        if(srcFlag == NORMAL_ARRAY && dstFlag == LSTM_ARRAY &&      //  Normalization-->LSTM
           normlayers[src].inputs() != lstmlayers[dst].inputDimensionality())
          return false;
        if(srcFlag == NORMAL_ARRAY && dstFlag == GRU_ARRAY &&       //  Normalization-->GRU
           normlayers[src].inputs() != grulayers[dst].inputDimensionality())
          return false;
        if(srcFlag == NORMAL_ARRAY && dstFlag == POOL_ARRAY &&      //  Normalization-->Pool
           normlayers[src].inputs() != poollayers[dst].width() * poollayers[dst].height())
          return false;
        if(srcFlag == NORMAL_ARRAY && dstFlag == UPRES_ARRAY &&     //  Normalization-->Upres
           normlayers[src].inputs() != upreslayers[dst].width() * upreslayers[dst].height())
          return false;
        if(srcFlag == NORMAL_ARRAY && dstFlag == NORMAL_ARRAY &&    //  Normalization-->Normalization
           normlayers[src].inputs() != normlayers[dst].inputs())
          return false;

        i = 0;                                                      //  Does this edge exist already?
        while(i < len && !( edgelist[i].srcType       == srcFlag       &&
                            edgelist[i].srcIndex      == src           &&
                            edgelist[i].selectorStart == selectorStart &&
                            edgelist[i].selectorEnd   == selectorEnd   &&
                            edgelist[i].dstType       == dstFlag       &&
                            edgelist[i].dstIndex      == dst           ))
          i++;
        if(i < len)
          return false;
                                                                    //  Check whether adding the proposed edge
        queue = new ArrayList<Node>();                              //  creates a cycle. Use DFS.
        queue.add(new Node(INPUT_ARRAY, 0));                        //  Enqueue the input Node.

        visited = new ArrayList<Node>();                            //  Initialize

        while(queue.size() > 0)
          {
            node = queue.remove(0);                                 //  Pop the first node from the queue

            i = 0;
            while(i < visited.size() && !(visited.get(i).type == node.type && visited.get(i).index == node.index))
              i++;
            if(i == visited.size())                                 //  Node has NOT been visited already
              {
                visited.add(node);                                  //  Mark the popped Node as visited
                                                                    //  Does the proposed link depart
                if(srcFlag == node.type && src == node.index)       //  from the popped Node?
                  {
                                                                    //  Does the proposed edge lead to a node
                    i = 0;                                          //  we've already visited?
                    while(i < visited.size() && !(visited.get(i).type == dstFlag && visited.get(i).index == dst))
                      i++;
                    if(i < visited.size())                          //  If so, then the proposed edge creates a cycle!
                      {
                        queue.clear();
                        visited.clear();
                        queue = null;
                        visited = null;
                        System.gc();                                //  Call the garbage collector

                        return false;
                      }

                    queue.add(new Node(dstFlag, dst));
                  }
                                                                    //  Find all existing connections
                for(i = 0; i < len; i++)                            //  from the popped Node, enqueue them.
                  {                                                 //  Enqueue a Node if it's reachable from 'node'
                    if(edgelist[i].srcType == node.type && edgelist[i].srcIndex == node.index)
                      {
                        j = 0;                                      //  Do we already have this connection?
                        while(j < queue.size() && !(queue.get(j).type  == edgelist[i].dstType &&
                                                    queue.get(j).index == edgelist[i].dstIndex))
                          j++;
                        if(j == queue.size())
                          queue.add(new Node(edgelist[i].dstType, edgelist[i].dstIndex));
                      }
                  }
              }
          }

        visited.clear();                                            //  Empty the list
        visited = null;                                             //  Release
        queue = null;

        if(len == 0)
          {
            edgelist = new Edge[1];
            edgelist[len] = new Edge(srcFlag, src, selectorStart, selectorEnd, dstFlag, dst);
          }
        else
          {
            tmp_edgelist = new Edge[len];                           //  Add the edge
            System.arraycopy(edgelist, 0, tmp_edgelist, 0, len);    //  Copy to temporary arrays
            edgelist = new Edge[len + 1];                           //  Re-allocate
            System.arraycopy(tmp_edgelist, 0, edgelist, 0, len);    //  Copy back into (expanded) original array
                                                                    //  Allocate another Edge in 'edgelist' array
            edgelist[len] = new Edge(srcFlag, src, selectorStart, selectorEnd, dstFlag, dst);
          }

        len++;                                                      //  Increment length of edgelist

        tmp_edgelist = null;                                        //  Force release of allocated memory
        System.gc();                                                //  Call the garbage collector

        return true;
      }

    /* Load a specific network from the file 'filename'. */
    public boolean load(String filename)
      {
        DataInputStream fp;
        byte byteBuffer[];
        byte srcType, dstType;
        int srcIndex, selectorStart, selectorEnd, dstIndex;
        int i, j;

        try
          {
            fp = new DataInputStream(new FileInputStream(filename));
          }
        catch(FileNotFoundException fileErr)
          {
            System.out.printf("ERROR: Unable to open %s.\n", filename);
            return false;
          }

        try
          {
            inputs = fp.readInt();                                  //  (int) Read NeuralNet input count from file
          }
        catch(IOException ioErr)
          {
            System.out.println("ERROR: Unable to read number of network inputs from file.");
            return false;
          }

        try
          {
            len = fp.readInt();
            if(len > 0)
              edgelist = new Edge[len];                             //  (int) Read NeuralNet edge count from file
          }
        catch(IOException ioErr)
          {
            System.out.println("ERROR: Unable to read number of network edges from file.");
            return false;
          }

        try
          {
            denseLen = fp.readInt();
            if(denseLen > 0)
              denselayers = new DenseLayer[denseLen];               //  (int) Read number of Dense Layers from file
          }
        catch(IOException ioErr)
          {
            System.out.println("ERROR: Unable to read number of dense layers from file.");
            return false;
          }

        try
          {
            convLen = fp.readInt();
            if(convLen > 0)
              convlayers = new Conv2DLayer[convLen];                //  (int) Read number of Convolutional Layers from file
          }
        catch(IOException ioErr)
          {
            System.out.println("ERROR: Unable to read number of convolutional layers from file.");
            return false;
          }

        try
          {
            accumLen = fp.readInt();
            if(accumLen > 0)
              accumlayers = new AccumLayer[accumLen];               //  (int) Read number of Accumulator Layers from file
          }
        catch(IOException ioErr)
          {
            System.out.println("ERROR: Unable to read number of accumulator layers from file.");
            return false;
          }

        try
          {
            lstmLen = fp.readInt();
            if(lstmLen > 0)
              lstmlayers = new LSTMLayer[lstmLen];                  //  (int) Read number of LSTM Layers from file
          }
        catch(IOException ioErr)
          {
            System.out.println("ERROR: Unable to read number of LSTM layers from file.");
            return false;
          }

        try
          {
            gruLen = fp.readInt();
            if(gruLen > 0)
              grulayers = new GRULayer[gruLen];                     //  (int) Read number of GRU Layers from file
          }
        catch(IOException ioErr)
          {
            System.out.println("ERROR: Unable to read number of GRU layers from file.");
            return false;
          }

        try
          {
            poolLen = fp.readInt();
            if(poolLen > 0)
              poollayers = new PoolLayer[poolLen];                  //  (int) Read number of Pool Layers from file
          }
        catch(IOException ioErr)
          {
            System.out.println("ERROR: Unable to read number of pooling layers from file.");
            return false;
          }

        try
          {
            upresLen = fp.readInt();
            if(upresLen > 0)
              upreslayers = new UpresLayer[upresLen];               //  (int) Read number of Upres Layers from file
          }
        catch(IOException ioErr)
          {
            System.out.println("ERROR: Unable to read number of up-resolution layers from file.");
            return false;
          }

        try
          {
            normalLen = fp.readInt();
            if(normalLen > 0)
              normlayers = new NormalLayer[normalLen];              //  (int) Read number of Normal Layers from file
          }
        catch(IOException ioErr)
          {
            System.out.println("ERROR: Unable to read number of normalization layers from file.");
            return false;
          }

        try
          {
            vars = (int)fp.readByte();
            if(vars > 0)
              variables = new Variable[vars];                       //  (char) Read number of variables from file
          }
        catch(IOException ioErr)
          {
            System.out.println("ERROR: Unable to read number of network variables from file.");
            return false;
          }

        try
          {
            gen = fp.readInt();                                     //  (int) Read generation/epoch from file
          }
        catch(IOException ioErr)
          {
            System.out.println("ERROR: Unable to read network generation/epoch from file.");
            return false;
          }

        try
          {
            fit = fp.readDouble();                                  //  (double) Read network fitness from file
          }
        catch(IOException ioErr)
          {
            System.out.println("ERROR: Unable to read network fitness from file.");
            return false;
          }

        byteBuffer = new byte[COMMSTR_LEN];                         //  Allocate byte array
        for(i = 0; i < COMMSTR_LEN; i++)                            //  Blank out buffer
          byteBuffer[i] = 0x00;
        for(i = 0; i < COMMSTR_LEN; i++)                            //  Read in full amount (can include NULLs)
          {
            try
              {
                byteBuffer[i] = fp.readByte();
              }
            catch(IOException ioErr)
              {
                System.out.println("ERROR: Unable to read network comment from file.");
                return false;
              }
          }
        comment = new String(byteBuffer, StandardCharsets.UTF_8);   //  Convert byte array to String

        for(i = 0; i < len; i++)                                    //  Read all Edges from file
          {
            try
              {
                srcType = fp.readByte();                            //  (char) Read edge source type from file
              }
            catch(IOException ioErr)
              {
                System.out.println("ERROR: Unable to read network edge list (source type byte) from file.");
                return false;
              }
            try
              {
                srcIndex = fp.readInt();                            //  (int) Read edge source index from file
              }
            catch(IOException ioErr)
              {
                System.out.println("ERROR: Unable to read network edge list (source index) from file.");
                return false;
              }
            try
              {
                selectorStart = fp.readInt();                       //  (int) Read edge selector start from file
              }
            catch(IOException ioErr)
              {
                System.out.println("ERROR: Unable to read network edge list (selector start) from file.");
                return false;
              }
            try
              {
                selectorEnd = fp.readInt();                         //  (int) Read edge selector end from file
              }
            catch(IOException ioErr)
              {
                System.out.println("ERROR: Unable to read network edge list (selector end) from file.");
                return false;
              }
            try
              {
                dstType = fp.readByte();                            //  (char) Read edge destination type from file
              }
            catch(IOException ioErr)
              {
                System.out.println("ERROR: Unable to read network edge list (destination type byte) from file.");
                return false;
              }
            try
              {
                dstIndex = fp.readInt();                            //  (int) Read edge destination index from file
              }
            catch(IOException ioErr)
              {
                System.out.println("ERROR: Unable to read network edge list (destination index) from file.");
                return false;
              }
            edgelist[i] = new Edge((int)srcType, srcIndex, selectorStart, selectorEnd, (int)dstType, dstIndex);
          }

        if(denseLen > 0)
          {
            for(i = 0; i < denseLen; i++)
              {
                denselayers[i] = new DenseLayer();                  //  Create one, just to read
                if(!denselayers[i].read(fp))
                  {
                    System.out.printf("ERROR: Failed to read network dense layer[%d]\n.", i);
                    return false;
                  }
              }
          }
        if(convLen > 0)
          {
            for(i = 0; i < convLen; i++)
              {
                convlayers[i] = new Conv2DLayer();                  //  Create one, just to read
                if(!convlayers[i].read(fp))
                  {
                    System.out.printf("ERROR: Failed to read network convolutional(2D) layer[%d]\n.", i);
                    return false;
                  }
              }
          }
        if(accumLen > 0)
          {
            for(i = 0; i < accumLen; i++)
              {
                accumlayers[i] = new AccumLayer();                  //  Create one, just to read
                if(!accumlayers[i].read(fp))
                  {
                    System.out.printf("ERROR: Failed to read network accumulation layer[%d]\n.", i);
                    return false;
                  }
              }
          }
        if(lstmLen > 0)
          {
            for(i = 0; i < lstmLen; i++)
              {
                lstmlayers[i] = new LSTMLayer();                    //  Create one, just to read
                if(!lstmlayers[i].read(fp))
                  {
                    System.out.printf("ERROR: Failed to read network LSTM layer[%d]\n.", i);
                    return false;
                  }
              }
          }
        if(gruLen > 0)
          {
            for(i = 0; i < gruLen; i++)
              {
                grulayers[i] = new GRULayer();                      //  Create one, just to read
                if(!grulayers[i].read(fp))
                  {
                    System.out.printf("ERROR: Failed to read network GRU layer[%d]\n.", i);
                    return false;
                  }
              }
          }
        if(poolLen > 0)
          {
            for(i = 0; i < poolLen; i++)
              {
                poollayers[i] = new PoolLayer();                    //  Create one, just to read
                if(!poollayers[i].read(fp))
                  {
                    System.out.printf("ERROR: Failed to read network 2D pooling layer[%d]\n.", i);
                    return false;
                  }
              }
          }
        if(upresLen > 0)
          {
            for(i = 0; i < upresLen; i++)
              {
                upreslayers[i] = new UpresLayer();                  //  Create one, just to read
                if(!upreslayers[i].read(fp))
                  {
                    System.out.printf("ERROR: Failed to read network up-resolution layer[%d]\n.", i);
                    return false;
                  }
              }
          }
        if(normalLen > 0)
          {
            for(i = 0; i < normalLen; i++)
              {
                normlayers[i] = new NormalLayer();                  //  Create one, just to read
                if(!normlayers[i].read(fp))
                  {
                    System.out.printf("ERROR: Failed to read network normalization layer[%d]\n.", i);
                    return false;
                  }
              }
          }

        byteBuffer = new byte[VARSTR_LEN];                          //  Re-allocate buffer
        for(i = 0; i < vars; i++)                                   //  Write all Variables to file
          {
            for(j = 0; j < VARSTR_LEN; j++)                         //  Blank out buffer
              byteBuffer[j] = 0x00;
            for(j = 0; j < VARSTR_LEN; j++)
              {
                try
                  {
                    byteBuffer[j] = fp.readByte();                  //  Read full length
                  }
                catch(IOException ioErr)
                  {
                    System.out.println("ERROR: Unable to read network variable key from file.");
                    return false;
                  }
              }
            variables[i] = new Variable();
                                                                    //  Convert byte array to String
            variables[i].key = new String(byteBuffer, StandardCharsets.UTF_8);
            try
              {
                variables[i].value = fp.readDouble();
              }
            catch(IOException ioErr)
              {
                System.out.println("ERROR: Unable to read network variable key from file.");
                return false;
              }
          }

        try
          {
            fp.close();
          }
        catch(IOException ioErr)
          {
            System.out.println("ERROR: Unable to close file.");
            return false;
          }

        byteBuffer = null;                                          //  Release array
        System.gc();                                                //  Summon the garbage collector

        return true;
      }

    /* Write the given Neural Network to a binary file named 'filename'. */
    public boolean write(String filename)
      {
        DataOutputStream fp;
        byte byteBuffer[];
        byte srcType, dstType;
        int srcIndex, selectorStart, selectorEnd, dstIndex;
        int i, j;

        try
          {
            fp = new DataOutputStream(new FileOutputStream(filename));
          }
        catch(FileNotFoundException fileErr)
          {
            System.out.printf("ERROR: Unable to create %s.\n", filename);
            return false;
          }

        try
          {
            fp.writeInt(inputs);                                    //  (int) Save NeuralNet input count to file
          }
        catch(IOException ioErr)
          {
            System.out.println("ERROR: Unable to write number of network inputs to file.");
            return false;
          }

        try
          {
            fp.writeInt(len);                                       //  (int) Save NeuralNet edge count to file
          }
        catch(IOException ioErr)
          {
            System.out.println("ERROR: Unable to write number of network edges to file.");
            return false;
          }

        try
          {
            fp.writeInt(denseLen);                                  //  (int) Save number of Dense Layers to file
          }
        catch(IOException ioErr)
          {
            System.out.println("ERROR: Unable to write number of network dense layers to file.");
            return false;
          }

        try
          {
            fp.writeInt(convLen);                                   //  (int) Save number of Convolutional Layers to file
          }
        catch(IOException ioErr)
          {
            System.out.println("ERROR: Unable to write number of network convolutional layers to file.");
            return false;
          }

        try
          {
            fp.writeInt(accumLen);                                  //  (int) Save number of Accumulator Layers to file
          }
        catch(IOException ioErr)
          {
            System.out.println("ERROR: Unable to write number of network accumulator layers to file.");
            return false;
          }

        try
          {
            fp.writeInt(lstmLen);                                   //  (int) Save number of LSTM Layers to file
          }
        catch(IOException ioErr)
          {
            System.out.println("ERROR: Unable to write number of network LSTM layers to file.");
            return false;
          }

        try
          {
            fp.writeInt(gruLen);                                    //  (int) Save number of GRU Layers to file
          }
        catch(IOException ioErr)
          {
            System.out.println("ERROR: Unable to write number of network GRU layers to file.");
            return false;
          }

        try
          {
            fp.writeInt(poolLen);                                   //  (int) Save number of Pool Layers to file
          }
        catch(IOException ioErr)
          {
            System.out.println("ERROR: Unable to write number of network pooling layers to file.");
            return false;
          }

        try
          {
            fp.writeInt(upresLen);                                  //  (int) Save number of Upres Layers to file
          }
        catch(IOException ioErr)
          {
            System.out.println("ERROR: Unable to write number of network up-resolution layers to file.");
            return false;
          }

        try
          {
            fp.writeInt(normalLen);                                 //  (int) Save number of Normal Layers to file
          }
        catch(IOException ioErr)
          {
            System.out.println("ERROR: Unable to write number of network normalization layers to file.");
            return false;
          }

        try
          {
            fp.write(vars);                                         //  (char) Save number of Variables to file
          }
        catch(IOException ioErr)
          {
            System.out.println("ERROR: Unable to write number of network variables to file.");
            return false;
          }

        try
          {
            fp.writeInt(gen);                                       //  (int) Save generation/epoch to file
          }
        catch(IOException ioErr)
          {
            System.out.println("ERROR: Unable to write network generation/epoch to file.");
            return false;
          }

        try
          {
            fp.writeDouble(fit);                                    //  (double) Save fitness to file
          }
        catch(IOException ioErr)
          {
            System.out.println("ERROR: Unable to write network fitness to file.");
            return false;
          }

        byteBuffer = new byte[COMMSTR_LEN];                         //  Allocate
        for(i = 0; i < COMMSTR_LEN; i++)                            //  Blank out buffer
          byteBuffer[i] = 0x00;
        i = 0;
        while(i < COMMSTR_LEN && i < comment.length())              //  Fill in up to limit
          {
            byteBuffer[i] = (byte)comment.codePointAt(i);
            i++;
          }
        for(i = 0; i < COMMSTR_LEN; i++)                            //  Write network comment to file
          {
            try
              {
                fp.write(byteBuffer[i]);
              }
            catch(IOException ioErr)
              {
                System.out.println("ERROR: Unable to write network comment to file.");
                return false;
              }
          }

        for(i = 0; i < len; i++)                                    //  Write all Edges to file
          {
            srcType = (byte)edgelist[i].srcType;                    //  (char) Save edge source type to file
            srcIndex = edgelist[i].srcIndex;                        //  (int) Save edge source index to file

            selectorStart = edgelist[i].selectorStart;              //  (int) Save edge selector start to file
            selectorEnd = edgelist[i].selectorEnd;                  //  (int) Save edge selector end to file

            dstType = (byte)edgelist[i].dstType;                    //  (char) Save edge destination type to file
            dstIndex = edgelist[i].dstIndex;                        //  (int) Save edge destination index to file

            try
              {
                fp.write(srcType);
              }
            catch(IOException ioErr)
              {
                System.out.println("ERROR: Unable to write network edge list (source type byte) to file.");
                return false;
              }
            try
              {
                fp.writeInt(srcIndex);
              }
            catch(IOException ioErr)
              {
                System.out.println("ERROR: Unable to write network edge list (source index) to file.");
                return false;
              }
            try
              {
                fp.writeInt(selectorStart);
              }
            catch(IOException ioErr)
              {
                System.out.println("ERROR: Unable to write network edge list (selector start) to file.");
                return false;
              }
            try
              {
                fp.writeInt(selectorEnd);
              }
            catch(IOException ioErr)
              {
                System.out.println("ERROR: Unable to write network edge list (selector end) to file.");
                return false;
              }
            try
              {
                fp.write(dstType);
              }
            catch(IOException ioErr)
              {
                System.out.println("ERROR: Unable to write network edge list (destination type byte) to file.");
                return false;
              }
            try
              {
                fp.writeInt(dstIndex);
              }
            catch(IOException ioErr)
              {
                System.out.println("ERROR: Unable to write network edge list (destination index) to file.");
                return false;
              }
          }

        if(denseLen > 0)
          {
            for(i = 0; i < denseLen; i++)
              {
                if(!denselayers[i].write(fp))
                  {
                    System.out.printf("ERROR: Failed to write network dense layer[%d]\n.", i);
                    return false;
                  }
              }
          }
        if(convLen > 0)
          {
            for(i = 0; i < convLen; i++)
              {
                if(!convlayers[i].write(fp))
                  {
                    System.out.printf("ERROR: Failed to write network convolutional(2D) layer[%d]\n.", i);
                    return false;
                  }
              }
          }
        if(accumLen > 0)
          {
            for(i = 0; i < accumLen; i++)
              {
                if(!accumlayers[i].write(fp))
                  {
                    System.out.printf("ERROR: Failed to write network accumulation layer[%d]\n.", i);
                    return false;
                  }
              }
          }
        if(lstmLen > 0)
          {
            for(i = 0; i < lstmLen; i++)
              {
                if(!lstmlayers[i].write(fp))
                  {
                    System.out.printf("ERROR: Failed to write network LSTM layer[%d]\n.", i);
                    return false;
                  }
              }
          }
        if(gruLen > 0)
          {
            for(i = 0; i < gruLen; i++)
              {
                if(!grulayers[i].write(fp))
                  {
                    System.out.printf("ERROR: Failed to write network GRU layer[%d]\n.", i);
                    return false;
                  }
              }
          }
        if(poolLen > 0)
          {
            for(i = 0; i < poolLen; i++)
              {
                if(!poollayers[i].write(fp))
                  {
                    System.out.printf("ERROR: Failed to write network 2D pooling layer[%d]\n.", i);
                    return false;
                  }
              }
          }
        if(upresLen > 0)
          {
            for(i = 0; i < upresLen; i++)
              {
                if(!upreslayers[i].write(fp))
                  {
                    System.out.printf("ERROR: Failed to write network up-resolution layer[%d]\n.", i);
                    return false;
                  }
              }
          }
        if(normalLen > 0)
          {
            for(i = 0; i < normalLen; i++)
              {
                if(!normlayers[i].write(fp))
                  {
                    System.out.printf("ERROR: Failed to write network normalization layer[%d]\n.", i);
                    return false;
                  }
              }
          }

        byteBuffer = new byte[VARSTR_LEN];                          //  Re-allocate array
        for(i = 0; i < vars; i++)                                   //  Write all Variables to file
          {
            for(j = 0; j < VARSTR_LEN; j++)                         //  Blank out array
              byteBuffer[j] = 0x00;
                                                                    //  Convert string to byte array
            byteBuffer = variables[i].key.getBytes(StandardCharsets.UTF_8);
            for(j = 0; j < VARSTR_LEN; j++)                         //  Limit variable key name
              {
                try
                  {
                    fp.write(byteBuffer[j]);
                  }
                catch(IOException ioErr)
                  {
                    System.out.println("ERROR: Unable to write network variable key to file.");
                    return false;
                  }
              }
            try
              {
                fp.writeDouble(variables[i].value);
              }
            catch(IOException ioErr)
              {
                System.out.println("ERROR: Unable to write network variable value to file.");
                return false;
              }
          }

        try
          {
            fp.close();
          }
        catch(IOException ioErr)
          {
            System.out.println("ERROR: Unable to close file.");
            return false;
          }

        byteBuffer = null;                                          //  Release array
        System.gc();                                                //  Summon the garbage collector

        return true;
      }

    /* Put the given network's edge-list in "computing order," from the input layer to the output layer. */
    public void sortEdges()
      {
        int i, j, k, l;
        Node nodelist[];                                            //  Find which network layers do not have outbounds
        int listlen;                                                //  Nodes are network layers, plus one input layer

        Edge newlist[];                                             //  The sorted version we will copy into 'edgelist'
        int ptr;

        ArrayList<Edge> addition;                                   //  Track latest addition to sorted list
        ArrayList<Edge> swap;                                       //  Hold Edge objects to be moved

        listlen = denseLen + convLen + accumLen + lstmLen + gruLen + poolLen + upresLen + normalLen + 1;
        nodelist = new Node[listlen];                               //  Allocate an array of Nodes (type, index)
        nodelist[0] = new Node(INPUT_ARRAY, 0);                     //  Set the first Node to the input layer

        j = 1;
        for(i = 0; i < denseLen; i++)                               //  Add all Dense Layers
          {
            nodelist[j] = new Node(DENSE_ARRAY, i);
            j++;
          }
        for(i = 0; i < convLen; i++)                                //  Add all Convolutional Layers
          {
            nodelist[j] = new Node(CONV2D_ARRAY, i);
            j++;
          }
        for(i = 0; i < accumLen; i++)                               //  Add all Accumulator Layers
          {
            nodelist[j] = new Node(ACCUM_ARRAY, i);
            j++;
          }
        for(i = 0; i < lstmLen; i++)                                //  Add all LSTM Layers
          {
            nodelist[j] = new Node(LSTM_ARRAY, i);
            j++;
          }
        for(i = 0; i < gruLen; i++)                                 //  Add all GRU Layers
          {
            nodelist[j] = new Node(GRU_ARRAY, i);
            j++;
          }
        for(i = 0; i < poolLen; i++)                                //  Add all Pool Layers
          {
            nodelist[j] = new Node(POOL_ARRAY, i);
            j++;
          }
        for(i = 0; i < upresLen; i++)                               //  Add all Upres Layers
          {
            nodelist[j] = new Node(UPRES_ARRAY, i);
            j++;
          }
        for(i = 0; i < normalLen; i++)                              //  Add all Normal Layers
          {
            nodelist[j] = new Node(NORMAL_ARRAY, i);
            j++;
          }

        //  By now we have an array of Nodes, one for each network layer, including the input layer.
        //  If we go through all network edges and cannot find an edge with a given layer as a source,
        //  then it is a network output

        swap = new ArrayList<Edge>();
        for(i = 0; i < listlen; i++)                                //  Go through all layers/nodes
          {
            j = 0;                                                  //  Try to find an outbound edge from this layer/node
            while(j < len && !(edgelist[j].srcType == nodelist[i].type &&
                               edgelist[j].srcIndex == nodelist[i].index))
              j++;
            if(j == len)                                            //  No outbound edge found:
              {                                                     //  this is a network output layer
                                                                    //  Add all edges that reach
                for(j = 0; j < len; j++)                            //    (nodelist[i].type, nodelist[i].index)
                  {
                    if(edgelist[j].dstType == nodelist[i].type && edgelist[j].dstIndex == nodelist[i].index)
                      swap.add(new Edge(edgelist[j].srcType, edgelist[j].srcIndex,
                                        edgelist[j].selectorStart, edgelist[j].selectorEnd,
                                        edgelist[j].dstType, edgelist[j].dstIndex));
                  }
              }
          }

        ptr = len - 1;                                              //  Point to new list's last cell
        newlist = new Edge[len];                                    //  Allocate new edge list

        addition = new ArrayList<Edge>();

        while(swap.size() > 0)                                      //  Loop until 'swap' comes up empty
          {
            for(i = 0; i < swap.size(); i++)                        //  Copy swap --> addition
              {                                                     //   and swap --> newlist at ptr
                addition.add(new Edge(swap.get(i).srcType, swap.get(i).srcIndex,
                                      swap.get(i).selectorStart, swap.get(i).selectorEnd,
                                      swap.get(i).dstType, swap.get(i).dstIndex));

                newlist[ptr - (swap.size() - 1) + i] = new Edge(swap.get(i).srcType, swap.get(i).srcIndex,
                                                                swap.get(i).selectorStart, swap.get(i).selectorEnd,
                                                                swap.get(i).dstType, swap.get(i).dstIndex);
              }

            ptr -= swap.size();                                     //  "Advance" pointer (toward head of array)
            swap.clear();                                           //  Empty 'swap'

            for(i = 0; i < addition.size(); i++)                    //  Scan over 'addition':
              {                                                     //  Find every edge that ends with a Node
                for(j = 0; j < len; j++)                            //  with which any member of 'addition' begins.
                  {
                    if( edgelist[j].dstType == addition.get(i).srcType &&
                        edgelist[j].dstIndex == addition.get(i).srcIndex )
                      {
                                                                    //  Is this Edge already in newlist,
                        k = ptr + 1;                                //  between ptr + 1 and the end of newlist?
                        while(k < newlist.length && !(edgelist[j].srcType == newlist[k].srcType &&
                                                      edgelist[j].srcIndex == newlist[k].srcIndex &&
                                                      edgelist[j].dstType == newlist[k].dstType &&
                                                      edgelist[j].dstIndex == newlist[k].dstIndex ))
                          k++;
                                                                    //  If so, pull it out of newlist
                        if(k < newlist.length)                      //  and close the gap in newlist
                          {
                            for(l = k; l >= 1; l--)
                              {
                                newlist[l].srcType = newlist[l - 1].srcType;
                                newlist[l].srcIndex = newlist[l - 1].srcIndex;
                                newlist[l].selectorStart = newlist[l - 1].selectorStart;
                                newlist[l].selectorEnd = newlist[l - 1].selectorEnd;
                                newlist[l].dstType = newlist[l - 1].dstType;
                                newlist[l].dstIndex = newlist[l - 1].dstIndex;
                              }
                            ptr++;                                  //  Move toward array tail
                                                                    //  for the single element we took out
                          }
                                                                    //  Add it to swap
                        swap.add(new Edge(edgelist[j].srcType, edgelist[j].srcIndex,
                                          edgelist[j].selectorStart, edgelist[j].selectorEnd,
                                          edgelist[j].dstType, edgelist[j].dstIndex));
                      }
                  }
              }

            addition.clear();                                       //  Empty array
          }

        for(i = 0; i < len; i++)                                    //  Sorting's complete:
          {                                                         //  write sorted edges to Neural Net
            edgelist[i].srcType = newlist[i].srcType;
            edgelist[i].srcIndex = newlist[i].srcIndex;
            edgelist[i].selectorStart = newlist[i].selectorStart;
            edgelist[i].selectorEnd = newlist[i].selectorEnd;
            edgelist[i].dstType = newlist[i].dstType;
            edgelist[i].dstIndex = newlist[i].dstIndex;
          }

        newlist = null;                                             //  Release arrays
        nodelist = null;
        System.gc();                                                //  Call the garbage collector

        return;
      }

    /* Find the network layer with the given 'name'.
       If it's found, we assume that YOU KNOW WHAT TYPE OF LAYER IT IS because all you'll get back is the array index.
       If it's not found, return Integer.MAX_VALUE.
       Results are undefined if there are more than one layer with the same name. */
    public int nameIndex(String name)
      {
        int i;

        i = 0;                                                      //  Check Dense layers
        while(i < denseLen && !denselayers[i].name().equals(name))
          i++;
        if(i < denseLen)
          return i;

        i = 0;                                                      //  Check Conv2D layers
        while(i < convLen && !convlayers[i].name().equals(name))
          i++;
        if(i < convLen)
          return i;

        i = 0;                                                      //  Check Accum layers
        while(i < accumLen && !accumlayers[i].name().equals(name))
          i++;
        if(i < accumLen)
          return i;

        i = 0;                                                      //  Check LSTM layers
        while(i < lstmLen && !lstmlayers[i].name().equals(name))
          i++;
        if(i < lstmLen)
          return i;

        i = 0;                                                      //  Check GRU layers
        while(i < gruLen && !grulayers[i].name().equals(name))
          i++;
        if(i < gruLen)
          return i;

        i = 0;                                                      //  Check Pool layers
        while(i < poolLen && !poollayers[i].name().equals(name))
          i++;
        if(i < poolLen)
          return i;

        i = 0;                                                      //  Check Upres layers
        while(i < upresLen && !upreslayers[i].name().equals(name))
          i++;
        if(i < upresLen)
          return i;

        i = 0;                                                      //  Check Normalization layers
        while(i < normalLen && !normlayers[i].name().equals(name))
          i++;
        if(i < normalLen)
          return i;

        return Integer.MAX_VALUE;
      }

    /* Find the type of layer with the given 'name'.
       If the layer is found, return one of the flags above indicating the array in which it was found.
       If the layer is NOT found, return Integer.MAX_VALUE.
       Results are undefined if there are more than one layer with the same name. */
    public int nameType(String name)
      {
        int i;

        i = 0;                                                      //  Check Dense layers
        while(i < denseLen && !denselayers[i].name().equals(name))
          i++;
        if(i < denseLen)
          return DENSE_ARRAY;

        i = 0;                                                      //  Check Conv2D layers
        while(i < convLen && !convlayers[i].name().equals(name))
          i++;
        if(i < convLen)
          return CONV2D_ARRAY;

        i = 0;                                                      //  Check Accum layers
        while(i < accumLen && !accumlayers[i].name().equals(name))
          i++;
        if(i < accumLen)
          return ACCUM_ARRAY;

        i = 0;                                                      //  Check LSTM layers
        while(i < lstmLen && !lstmlayers[i].name().equals(name))
          i++;
        if(i < lstmLen)
          return LSTM_ARRAY;

        i = 0;                                                      //  Check GRU layers
        while(i < gruLen && !grulayers[i].name().equals(name))
          i++;
        if(i < gruLen)
          return GRU_ARRAY;

        i = 0;                                                      //  Check Pool layers
        while(i < poolLen && !poollayers[i].name().equals(name))
          i++;
        if(i < poolLen)
          return POOL_ARRAY;

        i = 0;                                                      //  Check Upres layers
        while(i < upresLen && !upreslayers[i].name().equals(name))
          i++;
        if(i < upresLen)
          return UPRES_ARRAY;

        i = 0;                                                      //  Check Normalization layers
        while(i < normalLen && !normlayers[i].name().equals(name))
          i++;
        if(i < normalLen)
          return NORMAL_ARRAY;

        return Integer.MAX_VALUE;
      }

    /* Print a table of the network's edge list */
    public void printEdgeList()
      {
        int i;

        System.out.println("Src\t\t\t\tDst");
        System.out.println("Type\tIndex\tStart\tEnd\tType\tIndex");
        System.out.println("=================================================");
        for(i = 0; i < len; i++)
          System.out.printf("%d\t%d\t%d\t%d\t%d\t%d\n", edgelist[i].srcType,
                                                        edgelist[i].srcIndex,
                                                        edgelist[i].selectorStart,
                                                        edgelist[i].selectorEnd,
                                                        edgelist[i].dstType,
                                                        edgelist[i].dstIndex);
        return;
      }

    /* Print out a summary of the given network */
    public void print()
      {
        int i;
        int j;
        int k;
        int convparams;
        boolean firstInline;
        StringBuffer buffer;

        System.out.println("Layer (type)    Output    Params    IN         OUT");
        System.out.println("===========================================================");
        for(i = 0; i < denseLen; i++)                               //  Print all Dense layers
          {
            firstInline = true;
            j = 0;
            while(j < 9 && j < denselayers[i].name().length() && denselayers[i].name().codePointAt(j) != 0)
              {
                System.out.printf("%c", denselayers[i].name().charAt(j));
                j++;
              }
            while(j < 9)
              {
                System.out.print(" ");
                j++;
              }
            System.out.print("(Dns.) ");
                                                                    //  Print output length
            buffer = new StringBuffer(Integer.toString(denselayers[i].nodes()));
            j = 0;
            while(j < 10 && j < buffer.length())
              {
                System.out.printf("%c", buffer.charAt(j));
                j++;
              }
            while(j < 10)
              {
                System.out.print(" ");
                j++;
              }
                                                                    //  Print number of parameters
            buffer = new StringBuffer(Integer.toString((denselayers[i].inputs() + 1) * denselayers[i].nodes()));
            j = 0;
            while(j < 10 && j < buffer.length())
              {
                System.out.printf("%c", buffer.charAt(j));
                j++;
              }
            while(j < 10)
              {
                System.out.print(" ");
                j++;
              }

            for(k = 0; k < len; k++)                                //  Print inputs to this layer
              {
                if(edgelist[k].dstType == DENSE_ARRAY && edgelist[k].dstIndex == i)
                  {
                    if(!firstInline)
                      System.out.print("                                    ");
                    printLayerName(edgelist[k].srcType, edgelist[k].srcIndex);
                    firstInline = false;
                  }
              }
            for(k = 0; k < len; k++)                                //  Print outputs from this layer
              {
                if(edgelist[k].srcType == DENSE_ARRAY && edgelist[k].srcIndex == i)
                  {
                    System.out.print("                                               ");
                    printLayerName(edgelist[k].dstType, edgelist[k].dstIndex);
                  }
              }

            System.out.print("\n");
          }
        for(i = 0; i < convLen; i++)                                //  Print all Convolutional layers
          {
            firstInline = true;
            j = 0;
            while(j < 9 && j < convlayers[i].name().length() && convlayers[i].name().codePointAt(j) != 0)
              {
                System.out.printf("%c", convlayers[i].name().charAt(j));
                j++;
              }
            while(j < 9)
              {
                System.out.print(" ");
                j++;
              }
            System.out.print("(C2D.) ");
                                                                    //  Print output length
            buffer = new StringBuffer(Integer.toString(convlayers[i].outputLen()));
            j = 0;
            while(j < 10 && j < buffer.length())
              {
                System.out.printf("%c", buffer.charAt(j));
                j++;
              }
            while(j < 10)
              {
                System.out.print(" ");
                j++;
              }
                                                                    //  Print number of parameters
            convparams = 0;
            for(k = 0; k < convlayers[i].numFilters(); k++)
              convparams += convlayers[i].filter(k).w * convlayers[i].filter(k).h + 1;
            buffer = new StringBuffer(Integer.toString(convparams));
            j = 0;
            while(j < 10 && j < buffer.length())
              {
                System.out.printf("%c", buffer.charAt(j));
                j++;
              }
            while(j < 10)
              {
                System.out.print(" ");
                j++;
              }

            for(k = 0; k < len; k++)                                //  Print inputs to this layer
              {
                if(edgelist[k].dstType == CONV2D_ARRAY && edgelist[k].dstIndex == i)
                  {
                    if(!firstInline)
                      System.out.print("                                    ");
                    printLayerName(edgelist[k].srcType, edgelist[k].srcIndex);
                    firstInline = false;
                  }
              }
            for(k = 0; k < len; k++)                                //  Print outputs from this layer
              {
                if(edgelist[k].srcType == CONV2D_ARRAY && edgelist[k].srcIndex == i)
                  {
                    System.out.print("                                               ");
                    printLayerName(edgelist[k].dstType, edgelist[k].dstIndex);
                  }
              }

            System.out.print("\n");
          }
        for(i = 0; i < accumLen; i++)                               //  Print all Accumulator layers
          {
            firstInline = true;
            j = 0;
            while(j < 9 && j < accumlayers[i].name().length() && accumlayers[i].name().codePointAt(j) != 0)
              {
                System.out.printf("%c", accumlayers[i].name().charAt(j));
                j++;
              }
            while(j < 9)
              {
                System.out.print(" ");
                j++;
              }
            System.out.printf("(Acc.) ");
                                                                    //  Print output length
            buffer = new StringBuffer(Integer.toString(accumlayers[i].inputs()));
            j = 0;
            while(j < 10 && j < buffer.length())
              {
                System.out.printf("%c", buffer.charAt(j));
                j++;
              }
            while(j < 10)
              {
                System.out.printf(" ");
                j++;
              }
            buffer = new StringBuffer(Integer.toString(0));         //  Print number of parameters
            j = 0;
            while(j < 10 && j < buffer.length())
              {
                System.out.printf("%c", buffer.charAt(j));
                j++;
              }
            while(j < 10)
              {
                System.out.print(" ");
                j++;
              }

            for(k = 0; k < len; k++)                                //  Print inputs to this layer
              {
                if(edgelist[k].dstType == ACCUM_ARRAY && edgelist[k].dstIndex == i)
                  {
                    if(!firstInline)
                      System.out.print("                                    ");
                    printLayerName(edgelist[k].srcType, edgelist[k].srcIndex);
                    firstInline = false;
                  }
              }
            for(k = 0; k < len; k++)                                //  Print outputs from this layer
              {
                if(edgelist[k].srcType == ACCUM_ARRAY && edgelist[k].srcIndex == i)
                  {
                    System.out.print("                                               ");
                    printLayerName(edgelist[k].dstType, edgelist[k].dstIndex);
                  }
              }

            System.out.print("\n");
          }
        for(i = 0; i < lstmLen; i++)                                //  Print all LSTM layers
          {
            firstInline = true;
            j = 0;
            while(j < 9 && j < lstmlayers[i].name().length() && lstmlayers[i].name().codePointAt(j) != 0)
              {
                System.out.printf("%c", lstmlayers[i].name().charAt(j));
                j++;
              }
            while(j < 9)
              {
                System.out.print(" ");
                j++;
              }
            System.out.print("(LSTM) ");
                                                                    //  Print output length
            buffer = new StringBuffer(Integer.toString(lstmlayers[i].stateDimensionality()));
            j = 0;
            while(j < 10 && j < buffer.length())
              {
                System.out.printf("%c", buffer.charAt(j));
                j++;
              }
            while(j < 10)
              {
                System.out.print(" ");
                j++;
              }
                                                                    //  Print number of parameters
            buffer = new StringBuffer(Integer.toString(4 * lstmlayers[i].inputDimensionality() * lstmlayers[i].stateDimensionality() +
                                                       4 * lstmlayers[i].stateDimensionality() * lstmlayers[i].stateDimensionality() +
                                                       4 * lstmlayers[i].stateDimensionality()));
            j = 0;
            while(j < 10 && j < buffer.length())
              {
                System.out.printf("%c", buffer.charAt(j));
                j++;
              }
            while(j < 10)
              {
                System.out.print(" ");
                j++;
              }

            for(k = 0; k < len; k++)                                //  Print inputs to this layer
              {
                if(edgelist[k].dstType == LSTM_ARRAY && edgelist[k].dstIndex == i)
                  {
                    if(!firstInline)
                      System.out.print("                                    ");
                    printLayerName(edgelist[k].srcType, edgelist[k].srcIndex);
                    firstInline = false;
                  }
              }
            for(k = 0; k < len; k++)                                //  Print outputs from this layer
              {
                if(edgelist[k].srcType == LSTM_ARRAY && edgelist[k].srcIndex == i)
                  {
                    System.out.print("                                               ");
                    printLayerName(edgelist[k].dstType, edgelist[k].dstIndex);
                  }
              }

            System.out.print("\n");
          }
        for(i = 0; i < gruLen; i++)                                 //  Print all GRU layers
          {
            firstInline = true;
            j = 0;
            while(j < 9 && j < grulayers[i].name().length() && grulayers[i].name().codePointAt(j) != 0)
              {
                System.out.printf("%c", grulayers[i].name().charAt(j));
                j++;
              }
            while(j < 9)
              {
                System.out.print(" ");
                j++;
              }
            System.out.print("(GRU)  ");
                                                                    //  Print output length
            buffer = new StringBuffer(Integer.toString(grulayers[i].stateDimensionality()));
            j = 0;
            while(j < 10 && j < buffer.length())
              {
                System.out.printf("%c", buffer.charAt(j));
                j++;
              }
            while(j < 10)
              {
                System.out.print(" ");
                j++;
              }
                                                                    //  Print number of parameters
            buffer = new StringBuffer(Integer.toString(3 * grulayers[i].inputDimensionality() * grulayers[i].stateDimensionality() +
                                                       3 * grulayers[i].stateDimensionality() * grulayers[i].stateDimensionality() +
                                                       3 * grulayers[i].stateDimensionality()));
            j = 0;
            while(j < 10 && j < buffer.length())
              {
                System.out.printf("%c", buffer.charAt(j));
                j++;
              }
            while(j < 10)
              {
                System.out.print(" ");
                j++;
              }

            for(k = 0; k < len; k++)                                //  Print inputs to this layer
              {
                if(edgelist[k].dstType == GRU_ARRAY && edgelist[k].dstIndex == i)
                  {
                    if(!firstInline)
                      System.out.print("                                    ");
                    printLayerName(edgelist[k].srcType, edgelist[k].srcIndex);
                    firstInline = false;
                  }
              }
            for(k = 0; k < len; k++)                                //  Print outputs from this layer
              {
                if(edgelist[k].srcType == GRU_ARRAY && edgelist[k].srcIndex == i)
                  {
                    System.out.print("                                               ");
                    printLayerName(edgelist[k].dstType, edgelist[k].dstIndex);
                  }
              }

            System.out.print("\n");
          }
        for(i = 0; i < poolLen; i++)                                //  Print all Pool layers
          {
            firstInline = true;
            j = 0;
            while(j < 9 && j < poollayers[i].name().length() && poollayers[i].name().codePointAt(j) != 0)
              {
                System.out.printf("%c", poollayers[i].name().charAt(j));
                j++;
              }
            while(j < 9)
              {
                System.out.print(" ");
                j++;
              }
            System.out.print("(Pool) ");
                                                                    //  Print output length
            buffer = new StringBuffer(Integer.toString(poollayers[i].outputLen()));
            j = 0;
            while(j < 10 && j < buffer.length())
              {
                System.out.printf("%c", buffer.charAt(j));
                j++;
              }
            while(j < 10)
              {
                System.out.print(" ");
                j++;
              }
                                                                    //  Print number of parameters
            buffer = new StringBuffer(Integer.toString(0));
            j = 0;
            while(j < 10 && j < buffer.length())
              {
                System.out.printf("%c", buffer.charAt(j));
                j++;
              }
            while(j < 10)
              {
                System.out.print(" ");
                j++;
              }

            for(k = 0; k < len; k++)                                //  Print inputs to this layer
              {
                if(edgelist[k].dstType == POOL_ARRAY && edgelist[k].dstIndex == i)
                  {
                    if(!firstInline)
                      System.out.print("                                    ");
                    printLayerName(edgelist[k].srcType, edgelist[k].srcIndex);
                    firstInline = false;
                  }
              }
            for(k = 0; k < len; k++)                                //  Print outputs from this layer
              {
                if(edgelist[k].srcType == POOL_ARRAY && edgelist[k].srcIndex == i)
                  {
                    System.out.print("                                               ");
                    printLayerName(edgelist[k].dstType, edgelist[k].dstIndex);
                  }
              }

            System.out.print("\n");
          }
        for(i = 0; i < upresLen; i++)                               //  Print all Upres layers
          {
            firstInline = true;
            j = 0;
            while(j < 9 && j < upreslayers[i].name().length() && upreslayers[i].name().codePointAt(j) != 0)
              {
                System.out.printf("%c", upreslayers[i].name().charAt(j));
                j++;
              }
            while(j < 9)
              {
                System.out.print(" ");
                j++;
              }
            System.out.print("(Upres) ");
                                                                    //  Print output length
            buffer = new StringBuffer(Integer.toString(upreslayers[i].outputLen()));
            j = 0;
            while(j < 10 && j < buffer.length())
              {
                System.out.printf("%c", buffer.charAt(j));
                j++;
              }
            while(j < 10)
              {
                System.out.print(" ");
                j++;
              }
                                                                    //  Print number of parameters
            buffer = new StringBuffer(Integer.toString(0));
            j = 0;
            while(j < 10 && j < buffer.length())
              {
                System.out.printf("%c", buffer.charAt(j));
                j++;
              }
            while(j < 10)
              {
                System.out.print(" ");
                j++;
              }

            for(k = 0; k < len; k++)                                //  Print inputs to this layer
              {
                if(edgelist[k].dstType == UPRES_ARRAY && edgelist[k].dstIndex == i)
                  {
                    if(!firstInline)
                      System.out.print("                                    ");
                    printLayerName(edgelist[k].srcType, edgelist[k].srcIndex);
                    firstInline = false;
                  }
              }
            for(k = 0; k < len; k++)                                //  Print outputs from this layer
              {
                if(edgelist[k].srcType == UPRES_ARRAY && edgelist[k].srcIndex == i)
                  {
                    System.out.print("                                               ");
                    printLayerName(edgelist[k].dstType, edgelist[k].dstIndex);
                  }
              }

            System.out.print("\n");
          }
        for(i = 0; i < normalLen; i++)                              //  Print all Normalization layers
          {
            firstInline = true;
            j = 0;
            while(j < 9 && j < normlayers[i].name().length() && normlayers[i].name().codePointAt(j) != 0)
              {
                System.out.printf("%c", normlayers[i].name().charAt(j));
                j++;
              }
            while(j < 9)
              {
                System.out.print(" ");
                j++;
              }
            System.out.print("(Norm) ");
                                                                    //  Print output length
            buffer = new StringBuffer(Integer.toString(normlayers[i].inputs()));
            j = 0;
            while(j < 10 && j < buffer.length())
              {
                System.out.printf("%c", buffer.charAt(j));
                j++;
              }
            while(j < 10)
              {
                System.out.print(" ");
                j++;
              }
                                                                    //  Print number of parameters
            buffer = new StringBuffer(Integer.toString(4));
            j = 0;
            while(j < 10 && j < buffer.length())
              {
                System.out.printf("%c", buffer.charAt(j));
                j++;
              }
            while(j < 10)
              {
                System.out.print(" ");
                j++;
              }

            for(k = 0; k < len; k++)                                //  Print inputs to this layer
              {
                if(edgelist[k].dstType == NORMAL_ARRAY && edgelist[k].dstIndex == i)
                  {
                    if(!firstInline)
                      System.out.print("                                    ");
                    printLayerName(edgelist[k].srcType, edgelist[k].srcIndex);
                    firstInline = false;
                  }
              }
            for(k = 0; k < len; k++)                                //  Print outputs from this layer
              {
                if(edgelist[k].srcType == NORMAL_ARRAY && edgelist[k].srcIndex == i)
                  {
                    System.out.print("                                               ");
                    printLayerName(edgelist[k].dstType, edgelist[k].dstIndex);
                  }
              }

            System.out.print("\n");
          }

        System.out.println("===========================================================");
        return;
      }

    /* Called by print() */
    private void printLayerName(int arr, int index)
      {
        int i;

        switch(arr)
          {
            case INPUT_ARRAY:  System.out.print("NETWORK-IN");
                               break;
            case DENSE_ARRAY:  i = 0;
                               while(i < 9 && i < denselayers[ index ].name().length() && denselayers[ index ].name().codePointAt(i) != 0)
                                 {
                                   System.out.printf("%c", denselayers[ index ].name().charAt(i));
                                   i++;
                                 }
                               break;
            case CONV2D_ARRAY: i = 0;
                               while(i < 9 && i < convlayers[ index ].name().length() && convlayers[ index ].name().codePointAt(i) != 0)
                                 {
                                   System.out.printf("%c", convlayers[ index ].name().charAt(i));
                                   i++;
                                 }
                               break;
            case ACCUM_ARRAY:  i = 0;
                               while(i < 9 && i < accumlayers[ index ].name().length() && accumlayers[ index ].name().codePointAt(i) != 0)
                                 {
                                   System.out.printf("%c", accumlayers[ index ].name().charAt(i));
                                   i++;
                                 }
                               break;
            case LSTM_ARRAY:   i = 0;
                               while(i < 9 && i < lstmlayers[ index ].name().length() && lstmlayers[ index ].name().codePointAt(i) != 0)
                                 {
                                   System.out.printf("%c", lstmlayers[ index ].name().charAt(i));
                                   i++;
                                 }
                               break;
            case GRU_ARRAY:    i = 0;
                               while(i < 9 && i < grulayers[ index ].name().length() && grulayers[ index ].name().codePointAt(i) != 0)
                                 {
                                   System.out.printf("%c", grulayers[ index ].name().charAt(i));
                                   i++;
                                 }
                               break;
            case POOL_ARRAY:   i = 0;
                               while(i < 9 && i < poollayers[ index ].name().length() && poollayers[ index ].name().codePointAt(i) != 0)
                                 {
                                   System.out.printf("%c", poollayers[ index ].name().charAt(i));
                                   i++;
                                 }
                               break;
            case UPRES_ARRAY:  i = 0;
                               while(i < 9 && i < upreslayers[ index ].name().length() && upreslayers[ index ].name().codePointAt(i) != 0)
                                 {
                                   System.out.printf("%c", upreslayers[ index ].name().charAt(i));
                                   i++;
                                 }
                               break;
            case NORMAL_ARRAY: i = 0;
                               while(i < 9 && i < normlayers[ index ].name().length() && normlayers[ index ].name().codePointAt(i) != 0)
                                 {
                                   System.out.printf("%c", normlayers[ index ].name().charAt(i));
                                   i++;
                                 }
                               break;
          }

        System.out.print("\n");
        return;
      }

    /* Note that the comment gets cropped to 'COMMSTR_LEN' characters when written to file */
    public void setComment(String comm)
      {
        comment = new String(comm);
        return;
      }

    public String getComment()
      {
        return comment;
      }

    /****************************************************************
     Dense-Layers  */
    /* Add a layer to a network in progress.
       It shall have 'inputs' inputs and 'nodes' nodes. */
    public int addDense(int numInputs, int nodes)
      {
        DenseLayer tmp_denselayers[];

        if(denseLen == 0)
          {
            denselayers = new DenseLayer[1];                        //  Allocate DenseLayer in 'denselayers' array
            denselayers[0] = new DenseLayer(numInputs, nodes);
          }
        else
          {
            tmp_denselayers = new DenseLayer[denseLen];             //  Allocate temporary arrays
                                                                    //  Copy to temporary arrays
            System.arraycopy(denselayers, 0, tmp_denselayers, 0, denseLen);

            denselayers = new DenseLayer[denseLen + 1];             //  Re-allocate 'denselayers' array
                                                                    //  Copy back into (expanded) original arrays
            System.arraycopy(tmp_denselayers, 0, denselayers, 0, denseLen);
                                                                    //  Allocate another DenseLayer in 'denselayers' array
            denselayers[denseLen] = new DenseLayer(numInputs, nodes);

            tmp_denselayers = null;                                 //  Force release of allocated memory
            System.gc();                                            //  Call the garbage collector
          }

        denseLen++;
        return denseLen;
      }

    public DenseLayer dense(int index)
      {
        if(index < denseLen)
          return denselayers[index];
        return null;
      }

    /****************************************************************
     2D-Convolutional-Layers  */
    /* Add a Conv2DLayer to a network in progress.
       It shall have an 'inputW' by 'inputH' input matrix.
       Note that this function DOES NOT, itself, allocate any filters! */
    public int addConv2D(int inputW, int inputH)
      {
        Conv2DLayer tmp_convlayers[];

        if(convLen == 0)
          {
            convlayers = new Conv2DLayer[1];                        //  Allocate Conv2DLayer in 'convlayers' array
            convlayers[0] = new Conv2DLayer(inputW, inputH);
          }
        else
          {
            tmp_convlayers = new Conv2DLayer[convLen];              //  Allocate temporary arrays
                                                                    //  Copy to temporary arrays
            System.arraycopy(convlayers, 0, tmp_convlayers, 0, convLen);

            convlayers = new Conv2DLayer[convLen + 1];              //  Re-allocate 'convlayers' array
                                                                    //  Copy back into (expanded) original arrays
            System.arraycopy(tmp_convlayers, 0, convlayers, 0, convLen);

            convlayers[convLen] = new Conv2DLayer(inputW, inputH);  //  Allocate another Conv2DLayer in 'convlayers' array

            tmp_convlayers = null;                                  //  Force release of allocated memory
            System.gc();                                            //  Call the garbage collector
          }

        convLen++;
        return convLen;
      }

    public Conv2DLayer conv(int index)
      {
        if(index < convLen)
          return convlayers[index];
        return null;
      }

    /****************************************************************
     Accumulator-Layers  */
    /* Add an accumulator layer to a network in progress. */
    public int addAccum(int numInputs)
      {
        AccumLayer tmp_accumlayers[];

        if(accumLen == 0)
          {
            accumlayers = new AccumLayer[1];                        //  Allocate AccumLayer in 'accumlayers' array
            accumlayers[0] = new AccumLayer(numInputs);
          }
        else
          {
            tmp_accumlayers = new AccumLayer[accumLen];             //  Allocate temporary arrays
                                                                    //  Copy to temporary arrays
            System.arraycopy(accumlayers, 0, tmp_accumlayers, 0, accumLen);

            accumlayers = new AccumLayer[accumLen + 1];             //  Re-allocate 'accumlayers' array
                                                                    //  Copy back into (expanded) original arrays
            System.arraycopy(tmp_accumlayers, 0, accumlayers, 0, accumLen);

            accumlayers[accumLen] = new AccumLayer(numInputs);      //  Allocate another AccumLayer in 'accumlayers' array

            tmp_accumlayers = null;                                 //  Force release of allocated memory
            System.gc();                                            //  Call the garbage collector
          }

        accumLen++;
        return accumLen;
      }

    public AccumLayer accum(int index)
      {
        if(index < accumLen)
          return accumlayers[index];
        return null;
      }

    /****************************************************************
     LSTM-Layers  */
    /* Add an LSTM layer to a network in progress. */
    public int addLSTM(int dimInput, int dimState, int cacheSize)
      {
        LSTMLayer tmp_lstmlayers[];

        if(lstmLen == 0)
          {
            lstmlayers = new LSTMLayer[1];                          //  Allocate LSTMLayer in 'lstmlayers' array
            lstmlayers[0] = new LSTMLayer(dimInput, dimState, cacheSize);
          }
        else
          {
            tmp_lstmlayers = new LSTMLayer[lstmLen];                //  Allocate temporary arrays
                                                                    //  Copy to temporary arrays
            System.arraycopy(lstmlayers, 0, tmp_lstmlayers, 0, lstmLen);

            lstmlayers = new LSTMLayer[lstmLen + 1];                //  Re-allocate 'lstmlayers' array
                                                                    //  Copy back into (expanded) original arrays
            System.arraycopy(tmp_lstmlayers, 0, lstmlayers, 0, lstmLen);
                                                                    //  Allocate another LSTMLayer in 'lstmlayers' array
            lstmlayers[lstmLen] = new LSTMLayer(dimInput, dimState, cacheSize);

            tmp_lstmlayers = null;                                  //  Force release of allocated memory
            System.gc();                                            //  Call the garbage collector
          }

        lstmLen++;
        return lstmLen;
      }

    public LSTMLayer lstm(int index)
      {
        if(index < lstmLen)
          return lstmlayers[index];
        return null;
      }

    /****************************************************************
     GRU-Layers  */
    /* Add a GRU layer to a network in progress. */
    public int addGRU(int dimInput, int dimState, int cacheSize)
      {
        GRULayer tmp_grulayers[];

        if(gruLen == 0)
          {
            grulayers = new GRULayer[1];                            //  Allocate GRULayer in 'grulayers' array
            grulayers[0] = new GRULayer(dimInput, dimState, cacheSize);
          }
        else
          {
            tmp_grulayers = new GRULayer[gruLen];                   //  Allocate temporary arrays
                                                                    //  Copy to temporary arrays
            System.arraycopy(grulayers, 0, tmp_grulayers, 0, gruLen);

            grulayers = new GRULayer[gruLen + 1];                   //  Re-allocate 'grulayers' array
                                                                    //  Copy back into (expanded) original arrays
            System.arraycopy(tmp_grulayers, 0, grulayers, 0, gruLen);
                                                                    //  Allocate another GRULayer in 'grulayers' array
            grulayers[gruLen] = new GRULayer(dimInput, dimState, cacheSize);

            tmp_grulayers = null;                                   //  Force release of allocated memory
            System.gc();                                            //  Call the garbage collector
          }

        gruLen++;
        return gruLen;
      }

    public GRULayer gru(int index)
      {
        if(index < gruLen)
          return grulayers[index];
        return null;
      }

    /****************************************************************
     Pooling-Layers  */
    /* Add a PoolingLayer to a network in progress. */
    public int addPool(int inputW, int inputH)
      {
        PoolLayer tmp_poollayers[];

        if(poolLen == 0)
          {
            poollayers = new PoolLayer[1];                          //  Allocate PoolLayer in 'poollayers' array
            poollayers[0] = new PoolLayer(inputW, inputH);
          }
        else
          {
            tmp_poollayers = new PoolLayer[poolLen];                //  Allocate temporary arrays
                                                                    //  Copy to temporary arrays
            System.arraycopy(poollayers, 0, tmp_poollayers, 0, poolLen);

            poollayers = new PoolLayer[poolLen + 1];                //  Re-allocate 'poollayers' array
                                                                    //  Copy back into (expanded) original arrays
            System.arraycopy(tmp_poollayers, 0, poollayers, 0, poolLen);
            poollayers[poolLen] = new PoolLayer(inputW, inputH);    //  Allocate another PoolLayer in 'poollayers' array

            tmp_poollayers = null;                                  //  Force release of allocated memory
            System.gc();                                            //  Call the garbage collector
          }

        poolLen++;
        return poolLen;
      }

    public PoolLayer pooling(int index)
      {
        if(index < poolLen)
          return poollayers[index];
        return null;
      }

    /****************************************************************
     Upres-Layers  */
    /* Add a 2D upres layer to a network in progress. */
    public int addUpres(int inputW, int inputH)
      {
        UpresLayer tmp_upreslayers[];

        if(upresLen == 0)
          {
            upreslayers = new UpresLayer[1];                        //  Allocate UpresLayer in 'upreslayers' array
            upreslayers[0] = new UpresLayer(inputW, inputH);
          }
        else
          {
            tmp_upreslayers = new UpresLayer[upresLen];             //  Allocate temporary arrays
                                                                    //  Copy to temporary arrays
            System.arraycopy(upreslayers, 0, tmp_upreslayers, 0, upresLen);

            upreslayers = new UpresLayer[upresLen + 1];             //  Re-allocate 'upreslayers' array
                                                                    //  Copy back into (expanded) original arrays
            System.arraycopy(tmp_upreslayers, 0, upreslayers, 0, upresLen);
            upreslayers[upresLen] = new UpresLayer(inputW, inputH); //  Allocate another UpresLayer in 'upreslayers' array

            tmp_upreslayers = null;                                 //  Force release of allocated memory
            System.gc();                                            //  Call the garbage collector
          }

        upresLen++;
        return upresLen;
      }

    public UpresLayer upres(int index)
      {
        if(index < upresLen)
          return upreslayers[index];
        return null;
      }

    /****************************************************************
     Normalization-Layers  */
    /* Add a normalization layer to a network in progress. */
    public int addNormal(int numInputs)
      {
        NormalLayer tmp_normlayers[];

        if(normalLen == 0)
          {
            normlayers = new NormalLayer[1];                        //  Allocate NormalLayer in 'normlayers' array
            normlayers[0] = new NormalLayer(numInputs);
          }
        else
          {
            tmp_normlayers = new NormalLayer[normalLen];            //  Allocate temporary arrays
                                                                    //  Copy to temporary arrays
            System.arraycopy(normlayers, 0, tmp_normlayers, 0, normalLen);

            normlayers = new NormalLayer[normalLen + 1];            //  Re-allocate 'normlayers' array
                                                                    //  Copy back into (expanded) original arrays
            System.arraycopy(tmp_normlayers, 0, normlayers, 0, normalLen);
            normlayers[normalLen] = new NormalLayer(numInputs);     //  Allocate another NormalLayer in 'normlayers' array

            tmp_normlayers = null;                                  //  Force release of allocated memory
            System.gc();                                            //  Call the garbage collector
          }

        normalLen++;
        return normalLen;
      }

    public NormalLayer normal(int index)
      {
        if(index < normalLen)
          return normlayers[index];
        return null;
      }

    /****************************************************************
     Variable  */
    private class Variable
      {
        public String key;
        public double value;

        public Variable(String k, double v)
          {
            key = k;
            value = v;
          }

        public Variable()
          {
            key = "Var";
            value = 0.0;
          }
      }

    /****************************************************************
     Node  */
    private class Node
      {
        public int type;                                            //  Which network array to look in
        public int index;                                           //  Index into that array

        public Node(int t, int i)
          {
            type = t;
            index = i;
          }
      }

    /****************************************************************
     Edge  */
    private class Edge
      {
        public int srcType;                                         //  Indicates in which array to find the source
        public int srcIndex;                                        //  Index into that array

        public int selectorStart;                                   //  From (and including) this array element...
        public int selectorEnd;                                     //  ...to (but excluding) this array element.

        public int dstType;                                         //  Indicates in which array to find the destination
        public int dstIndex;                                        //  Index into that array

        public Edge(int srcT, int srcI, int selStart, int selEnd, int dstT, int dstI)
          {
            srcType = srcT;
            srcIndex = srcI;

            selectorStart = selStart;
            selectorEnd = selEnd;

            dstType = dstT;
            dstIndex = dstI;
          }
      }
  }
