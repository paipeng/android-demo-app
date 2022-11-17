package org.pytorch.helloworld;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.util.Log;
import android.widget.ImageView;
import android.widget.TextView;

import androidx.appcompat.app.AppCompatActivity;

import org.pytorch.IValue;
import org.pytorch.MemoryFormat;
import org.pytorch.Module;
import org.pytorch.Tensor;
import org.pytorch.torchvision.TensorImageUtils;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.FloatBuffer;

public class MainActivity extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        Bitmap bitmap = null;
        Module module = null;
        try {
            // creating bitmap from packaged into app android asset 'image.jpg',
            // app/src/main/assets/image.jpg
            bitmap = BitmapFactory.decodeStream(getAssets().open("board_2117_piece_3-2.bmp"));
            // loading serialized torchscript module from packaged into app android asset model.pt,
            // app/src/model/assets/model.pt
            //module = LiteModuleLoader.load(assetFilePath(this, "mobile_chess_model.ptl"));
            final String moduleFileAbsoluteFilePath = new File(
                    assetFilePath(this, "mobile_chess_model4.ptl")).getAbsolutePath();

            module = Module.load(moduleFileAbsoluteFilePath);
        } catch (IOException e) {
            Log.e("PytorchHelloWorld", "Error reading assets", e);
            finish();
        } catch (Exception e) {
            Log.e("PytorchHelloWorld", "Error reading assets", e);
            finish();
        }

        // showing image on UI
        ImageView imageView = findViewById(R.id.image);
        imageView.setImageBitmap(bitmap);

        // preparing input tensor
        final Tensor inputTensor = bitmapToGrayFloat32Tensor(bitmap,
                TensorImageUtils.TORCHVISION_NORM_MEAN_RGB, TensorImageUtils.TORCHVISION_NORM_STD_RGB, MemoryFormat.CONTIGUOUS);

        // running the model
        final Tensor outputTensor = module.forward(IValue.from(inputTensor)).toTensor();

        // getting tensor content as java array of floats
        final float[] scores = outputTensor.getDataAsFloatArray();

        // searching for the index with maximum score
        float maxScore = -Float.MAX_VALUE;
        int maxScoreIdx = -1;
        for (int i = 0; i < scores.length; i++) {
            if (scores[i] > maxScore) {
                maxScore = scores[i];
                maxScoreIdx = i;
            }
            Log.d("SCORE", "index: " + i + "  score: " + scores[i]);
        }


        String className = ChessNetClasses.IMAGENET_CLASSES[maxScoreIdx];

        // showing className on UI
        TextView textView = findViewById(R.id.text);
        textView.setText(className);
    }

    /**
     * Copies specified asset to the file in /files app directory and returns this file absolute path.
     *
     * @return absolute file path
     */
    public static String assetFilePath(Context context, String assetName) throws IOException {
        File file = new File(context.getFilesDir(), assetName);
        if (file.exists() && file.length() > 0) {
            return file.getAbsolutePath();
        }

        try (InputStream is = context.getAssets().open(assetName)) {
            try (OutputStream os = new FileOutputStream(file)) {
                byte[] buffer = new byte[4 * 1024];
                int read;
                while ((read = is.read(buffer)) != -1) {
                    os.write(buffer, 0, read);
                }
                os.flush();
            }
            return file.getAbsolutePath();
        }
    }


    public static Tensor bitmapToGrayFloat32Tensor(
            final Bitmap bitmap,
            float[] normMeanRGB,
            float[] normStdRGB,
            MemoryFormat memoryFormat) {
        final FloatBuffer floatBuffer = Tensor.allocateFloatBuffer(bitmap.getWidth() * bitmap.getHeight());
        bitmapToFloatBuffer(
                bitmap, 0, 0, bitmap.getWidth(), bitmap.getHeight(), normMeanRGB, normStdRGB, floatBuffer, 0, memoryFormat);
        return Tensor.fromBlob(floatBuffer, new long[]{1, 1, bitmap.getHeight(), bitmap.getWidth()}, memoryFormat);
    }

    public static Tensor bitmapToGrayFloat32Tensor(
            final Bitmap bitmap,
            int x,
            int y,
            int width,
            int height,
            float[] normMeanRGB,
            float[] normStdRGB,
            MemoryFormat memoryFormat) {


        final FloatBuffer floatBuffer = Tensor.allocateFloatBuffer(width * height);
        bitmapToFloatBuffer(
                bitmap, x, y, width, height, normMeanRGB, normStdRGB, floatBuffer, 0, memoryFormat);
        return Tensor.fromBlob(floatBuffer, new long[]{1, 1, height, width}, memoryFormat);
    }

    public static void bitmapToFloatBuffer(
            final Bitmap bitmap,
            final int x,
            final int y,
            final int width,
            final int height,
            final float[] normMeanRGB,
            final float[] normStdRGB,
            final FloatBuffer outBuffer,
            final int outBufferOffset,
            final MemoryFormat memoryFormat) {

        if (memoryFormat != MemoryFormat.CONTIGUOUS && memoryFormat != MemoryFormat.CHANNELS_LAST) {
            throw new IllegalArgumentException("Unsupported memory format " + memoryFormat);
        }

        final int pixelsCount = height * width;
        final int[] pixels = new int[pixelsCount];
        bitmap.getPixels(pixels, 0, width, x, y, width, height);
        if (MemoryFormat.CONTIGUOUS == memoryFormat) {
            for (int i = 0; i < pixelsCount; i++) {
                final int c = pixels[i];
                float r = (c & 0xff) / 255.0f;
                outBuffer.put(outBufferOffset + i, r);
            }
        } else {
            for (int i = 0; i < pixelsCount; i++) {
                final int c = pixels[i];
                float r = (c & 0xff) / 255.0f;
                outBuffer.put(outBufferOffset + i, r);
            }
        }
    }
}
