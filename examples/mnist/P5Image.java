import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.nio.file.Paths;

public class P5Image
  {
    public int w;                                                   //  Width of the image
    public int h;                                                   //  Height of the image

    public int maxGray;                                             //  Maximum grayscale value
    public int buffer[];                                            //  Image data

    public P5Image()
      {
        w = 28;                                                     //  Default to 28 x 28
        h = 28;
        maxGray = 255;                                              //  Default to 255
      }

    public int read(String filename)
      {
        Path imageFilePath;
        FileChannel imageFileChannel;

        DataInputStream fp;
        ByteBuffer byteBuffer;
        byte byteArr[];

        String str;
        int totalFileSize, linectr;
        int ptrs[] = {-1, -1, -1, -1};
        int reading = 0;
        int byteCtr;
        int i, j;

        imageFilePath = Paths.get(filename);
        try
          {
            imageFileChannel = FileChannel.open(imageFilePath);
          }
        catch(IOException ioErr)
          {
            System.out.printf("ERROR: Unable to open %s.\n", filename);
            return 0;
          }
        try
          {
            totalFileSize = (int)imageFileChannel.size();
          }
        catch(IOException ioErr)
          {
            System.out.printf("ERROR: Unable to get size for %s.\n", filename);
            return 0;
          }
        try
          {
            imageFileChannel.close();
          }
        catch(IOException ioErr)
          {
            System.out.printf("ERROR: Unable to close %s.\n", filename);
            return 0;
          }

        try                                                         //  Attempt to open
          {
            fp = new DataInputStream(new FileInputStream(filename));
          }
        catch(FileNotFoundException fileErr)
          {
            System.out.printf("ERROR: Unable to open %s.\n", filename);
            return 0;
          }

        byteArr = new byte[totalFileSize];                          //  Allocate space for the whole file

        try                                                         //  Attempt to read the file as a byte array
          {
            fp.read(byteArr);
          }
        catch(IOException ioErr)
          {
            System.out.println("ERROR: Unable to read file into byte array.");
            return 0;
          }

        byteBuffer = ByteBuffer.allocate(totalFileSize);
        byteBuffer = ByteBuffer.wrap(byteArr);
        byteBuffer.order(ByteOrder.LITTLE_ENDIAN);                  //  Read little-endian

        byteArr = new byte[3];
        for(i = 0; i < 3; i++)
          byteArr[i] = byteBuffer.get();
        str = new String(byteArr, StandardCharsets.UTF_8);
        if(!str.equals("P5\n"))
          {
            System.out.printf("ERROR: %s is not a P5 image.\n", filename);
            try
              {
                fp.close();
              }
            catch(IOException ioErr)
              {
                System.out.println("ERROR: Unable to close file.");
              }
            return 0;
          }

        linectr = 2;                                                //  We must read *at least* 2 more lines before binary data begin
        byteCtr = 3;                                                //  After "P5\n" we are three bytes into the file
        byteArr = new byte[1];                                      //  Reallocate this array for reading one character at a time
        while(linectr > 0)
          {
            byteArr[0] = byteBuffer.get();                          //  Is the first character of a new line #?
            byteCtr++;

            if(byteArr[0] == (byte)35)                              //  Yes: then this is a comment, not a real line.
              {
                while(byteArr[0] != (byte)10)                       //  Advance file pointer until carriage return and do not count this carriage return
                  {
                    byteArr[0] = byteBuffer.get();
                    byteCtr++;
                  }
              }
            else                                                    //  No: then this is a real line.
              {
                if(linectr == 2)                                    //  Reading width and height
                  {
                    ptrs[reading] = byteCtr - 1;                    //  Save the offset (inclusive) for the beginning of the image dimensions
                    reading++;

                    while(byteArr[0] != (byte)10)
                      {
                        byteArr[0] = byteBuffer.get();
                        byteCtr++;
                      }

                    ptrs[reading] = byteCtr - 1;                    //  Save the offset (exclusive) for the end of the image dimensions
                    reading++;
                  }
                else                                                //  Reading maximum gray value
                  {
                    ptrs[reading] = byteCtr - 1;                    //  Save the offset (inclusive) for the beginning of the maximum gray
                    reading++;

                    while(byteArr[0] != (byte)10)
                      {
                        byteArr[0] = byteBuffer.get();
                        byteCtr++;
                      }

                    ptrs[reading] = byteCtr - 1;                    //  Save the offset (exclusive) for the end of the maximum gray
                    reading++;
                  }
                linectr--;                                          //  Count this carriage return
              }
          }

        byteBuffer.rewind();                                        //  Build the substring containing the image dimensions
        byteArr = new byte[ptrs[1] - ptrs[0]];
        j = 0;
        for(i = ptrs[0]; i < ptrs[1]; i++)
          {
            byteArr[j] = byteBuffer.get(i);
            j++;
          }
        str = new String(byteArr, StandardCharsets.UTF_8);
        i = 0;
        while(str.charAt(i) != ' ')
          i++;
        w = Integer.parseInt(str.substring(0, i));                  //  FINALLY... the width
        h = Integer.parseInt(str.substring(i + 1));                 //        and the height

        byteBuffer.rewind();                                        //  Build the substring containing the maximum grayscale value
        byteArr = new byte[ptrs[3] - ptrs[2]];
        j = 0;
        for(i = ptrs[2]; i < ptrs[3]; i++)
          {
            byteArr[j] = byteBuffer.get(i);
            j++;
          }
        str = new String(byteArr, StandardCharsets.UTF_8);
        maxGray = Integer.parseInt(str);                            //  FINALLY... the maximum grayscale value

        buffer = new int[w * h];                                    //  Allocate
        byteBuffer.rewind();
        j = 0;
        for(i = ptrs[3] + 1; i < totalFileSize; i++)
          {
            buffer[j] = (int)byteBuffer.get(i) & 0xFF;
            j++;
          }

        try                                                         //  Attempt to close
          {
            fp.close();
          }
        catch(IOException ioErr)
          {
            System.out.println("ERROR: Unable to close file.");
            return 0;
          }
        return buffer.length;
      }
  }
