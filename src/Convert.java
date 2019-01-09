import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.*;

public class Convert {  // Класс конвертации изображения в массив байтов
    void images() throws IOException {
        byte[] imageInByte;
        BufferedImage originalImage = null;
        try {
            originalImage = ImageIO.read(new File("res/in.bmp"));
        } catch (IOException e) {
            e.printStackTrace();
        }

        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        try {
            ImageIO.write(originalImage, "bmp", baos);
        } catch (IOException e) {
            e.printStackTrace();
        }
        baos.flush();
        imageInByte = baos.toByteArray();
        baos.close();

        FileOutputStream txt = new FileOutputStream("res/ch.svg"); //запись массива байтов в файл
        txt.write(imageInByte);
        txt.close();
    }

}
