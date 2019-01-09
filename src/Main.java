import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

public class Main {
    public static void main(String[] args){
        List<List<Data>> data;
        Scanner scan = new Scanner(System.in);
        boolean flag = false;
        Loader loader = new Loader();
        data = loader.loadAllData();  //Загружаем данные из баз для обучения и тестирования
        List<Data> trainingData = data.get(0);
        List<Data> testData = data.get(1);
        System.out.println("Finished loading Data");
        int[] sizes = {784, 30, 10}; //Задаем размеры матрицы
        Network n = new Network(sizes); //Создаем объект нейросети
        try {
            loader.loadCode(2);
        } catch (IOException e) {
            e.printStackTrace();
        }

        n.SGD(trainingData, testData, 100, 50, 0.5);  //Используем функцию обучения нейросети



        while(!flag){  //Цикл, позволяющий отдавать команды и взаимодействовать с нейросетью
            System.out.println("");
            System.out.println("");
            System.out.println("Распознать цифру с рисунка - r. Распознать случайную цифру - g. Статистический тест(выборка 10 000) - t. Дополнительно обучить нейросеть - l. Выход - e");
            System.out.println("Введите команду : ");
            String f = scan.nextLine();
            System.out.println("");
            System.out.println("");
            if(f.equals("e")){
                flag = true;
            }
            if(f.equals("g")){
                int ran = n.check(testData);
                try {
                    loader.loadCode(ran);
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
            if(f.equals("t")){
                System.out.println("Successful: " + n.evaluate(testData) + "/" + 10000);
            }
            if(f.equals("l")){
                System.out.println("Введите число эпох: ");
                int k = scan.nextInt();
                n.SGD(trainingData, testData, k, 50, 0.5);
            }

            if(f.equals("r")){
                Convert convert = new Convert();
                try {
                    convert.images();
                } catch (IOException e) {
                    e.printStackTrace();
                }
                List<Data> paint = new ArrayList<>();
                try {
                    paint = loader.loadPrintData("res/ch.svg", "res/chb2.svg");
                } catch (IOException e) {
                    e.printStackTrace();
                }
                n.checkImg(paint);

            }
        }
    }
}