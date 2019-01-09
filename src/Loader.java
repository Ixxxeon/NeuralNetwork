import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.*;
import java.math.BigInteger;
import java.util.ArrayList;
import java.util.List;


public class Loader { //Класс загрузки информации из файлов(базы и картинок)
    static String trainingFile = "sets/train-images";
    static String trainingLabel = "sets/train-labels";
    static String testFile = "sets/test-images";
    static String testLabel = "sets/test-labels";

    int readInt(InputStream in) throws IOException{
        int d;
        int[] b = new int[4];
        for(int i = 0; i < 4; i++)
            b[i] = in.read();
        d = b[3] | b[2] << 8 | b[1] << 16 | b[0] << 24;
        return d;
    }

    List<Data> readDataFiles(String imageFile, String labelFile) throws IOException{  // Считывание информации для обучения нейросети из базы
        List<Data> dataList = new ArrayList<>();
        int[] imageData;
        int[] labelData;
        int totalRows, totalCols, totalImages, totalLabels;
        try(InputStream in = new FileInputStream(imageFile)){
            int magic = readInt(in);
            totalImages = readInt(in);
            totalRows = readInt(in);
            totalCols = readInt(in);
            imageData = new int[totalImages * totalRows * totalCols];
            for(int i = 0; i < totalImages * totalRows * totalCols; i++) {
                imageData[i] = in.read();
            }
            in.close();
        }

        try(InputStream in = new FileInputStream(labelFile)){
            int magic = readInt(in);
            totalLabels = readInt(in);
            labelData = new int[totalLabels];
            for(int i = 0; i < totalLabels; i++)
                labelData[i] = in.read();
            in.close();
        }
        if (totalImages != totalLabels)
            return null;
        int ic = 0;
        int lc = 0;
        while(ic < imageData.length && lc < labelData.length){
            Matrix input, result;
            input = new Matrix(totalRows * totalCols, 1);
            for(int i = 0; i < totalRows * totalCols; i++)
                input.set(i, 0, imageData[ic++]);
            result = new Matrix(10, 1);
            result.applyFunc(p -> 0.0);
            result.set(labelData[lc++], 0, 1.0);
            dataList.add(new Data(input, result));
        }
        return dataList;
    }

    List<Data> loadData(String imageFile, String labelFile){ //Загрузка информации в лист
        List<Data> dataList;
        try {
            dataList = readDataFiles(imageFile, labelFile);
        }catch(java.io.IOException e){
            System.out.println(e);
            dataList = null;
        }
        if(dataList == null)
            System.out.println("dataList null");
        return dataList;
    }

    List<Data> loadPrintData(String imageFile, String labelFile) throws IOException { //Считывание информации с массива байтов, взятого с картинки, нарисованной пользователем
        List<Data> dataList = new ArrayList<>();
        int[] imageData;
        int[] labelData = {0, 0};
        int totalRows = 28;
        int totalCols = 28;
        try(InputStream in = new FileInputStream(imageFile)){
            imageData = new int[totalRows * totalCols];
            for(int i = 0; i <  1078; i++) {
                in.read();
            }

            for(int i = 0; i <  totalRows * totalCols; i++) {
                imageData[i] = in.read();
            }
            in.close();
        }
        try(InputStream in = new FileInputStream(labelFile)){
            int magic = readInt(in);
            labelData[0] = in.read();
            in.close();
        }
        int ic = 0;
        int lc = 0;
        while(ic < imageData.length && lc < 1){
            Matrix input, result;
            input = new Matrix(totalRows * totalCols, 1);
            for(int i = 0; i < totalRows * totalCols; i++)
                input.set(i, 0, imageData[ic++]);
            result = new Matrix(10, 1);
            result.applyFunc(p -> 0.0);
            result.set(labelData[lc++], 0, 1.0);
            dataList.add(new Data(input, result));
        }

        return dataList;
    }


    void loadCode(int r) throws IOException { //Визуализация случайной цифры из базы
        byte[] ByteCode = new byte[1862];
        try(InputStream in1 = new FileInputStream("res/ch.svg")) {
            for (int i = 0; i < 1078; i++) {
                ByteCode[i] = (byte) in1.read();
            }
            in1.close();
        }
        try(InputStream in = new FileInputStream(testFile)){
            readInt(in);
            readInt(in);
            readInt(in);
            readInt(in);
            for(int i = 0; i <  784*r; i++) {
                in.read();
            }
            for(int i = 0; i < 784; i++) {
                ByteCode[i+1078] = (byte) in.read();
            }
            in.close();
        }
        InputStream inn = new ByteArrayInputStream(ByteCode);
        BufferedImage bImageFromConvert = ImageIO.read(inn);
        ImageIO.write(bImageFromConvert, "bmp", new File("res/out.bmp"));
        FileOutputStream txt = new FileOutputStream("res/ch1.svg"); //запись массива байтов в файл1
        txt.write(ByteCode);
        txt.close();
        inn.close();
    }




    List<List<Data>> loadAllData(){
        List<Data> trainingData = loadData(trainingFile, trainingLabel);
        List<Data> testData = loadData(testFile, testLabel);
        List<List<Data>> data = new ArrayList<>();
        data.add(trainingData);
        data.add(testData);
        return data;
    }



}

