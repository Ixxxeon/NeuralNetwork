
import java.util.*;



public class Network { // Класс нейросети
    private int totalLayers;
    Matrix[] biases;  //Массив весов
    Matrix[] weights;  //Массив весов

    Network(int[] sizes){  //Конструктор, создающий нейросеть нужного размера и заполняющий матрицы случайными весами
        totalLayers = sizes.length;
        biases = new Matrix[totalLayers - 1];
        weights = new Matrix[totalLayers - 1];
        for(int i = 0; i < totalLayers - 1; i++){
            Random rand = new Random();
            biases[i] = new Matrix(sizes[i + 1], 1);
            weights[i] = new Matrix(sizes[i + 1], sizes[i]);
            biases[i] = biases[i].applyFunc(p -> rand.nextGaussian());
            weights[i] = weights[i].applyFunc(p -> rand.nextGaussian());
        }
    }

    Matrix sigmoid(Matrix z){
        return z.applyFunc(p -> 1 / (1 + Math.pow(Math.E, -p)));
    } //Метод сигмоидной функции

    Matrix sigmoidPrime(Matrix z){
        Matrix x = sigmoid(z);
        x = x.applyFunc(p -> 1 - p);
        return sigmoid(z).schurProduct(x);
    }

    Matrix feedForward(Matrix inp){
        Matrix a = new Matrix(inp);
        for(int i = 0; i < totalLayers - 1; i++){
            a = sigmoid(weights[i].matrixMult(a).matrixAdd(biases[i]));
        }
        return a;
    }

    Matrix cost_derivative(Matrix output_activations, Matrix y){
        return output_activations.matrixSub(y);
    }

    List<Matrix[]> backprop(Data data){ //Вычисляет наблы и заносит их в лист
        Matrix[] nabla_b = new Matrix[totalLayers - 1];
        Matrix[] nabla_w = new Matrix[totalLayers - 1];
        for(int j = 0; j < totalLayers - 1; j++){
            nabla_b[j] = new Matrix(biases[j].get_rows(), biases[j].get_cols());
            nabla_w[j] = new Matrix(weights[j].get_rows(), weights[j].get_cols());
        }
        Matrix activation = new Matrix(data.input);
        List<Matrix> activations = new ArrayList<>();
        activations.add(activation);
        List<Matrix> zVector = new ArrayList<>();
        for(int j = 0; j < totalLayers - 1; j++){
            Matrix z = weights[j].matrixMult(activation).matrixAdd(biases[j]);
            zVector.add(z);
            activation = sigmoid(z);
            activations.add(activation);
        }
        Matrix delta = cost_derivative(activations.get(activations.size() - 1), data.result).schurProduct(sigmoidPrime(zVector.get(zVector.size() - 1)));
        nabla_b[nabla_b.length - 1] = delta;
        nabla_w[nabla_w.length - 1] = delta.matrixMult(activations.get(activations.size() - 2).matrixTranspose());
        for(int j = nabla_b.length - 2; j >= 0; j--){
            Matrix z = zVector.get(j);
            Matrix sp = sigmoidPrime(z);
            delta = weights[j + 1].matrixTranspose().matrixMult(delta).schurProduct(sp);
            nabla_b[j] = delta;
            nabla_w[j] = delta.matrixMult(activations.get(j).matrixTranspose());
        }
        List<Matrix[]> ret = new ArrayList<>();
        ret.add(nabla_b);
        ret.add(nabla_w);
        return ret;
    }

    void update_mini_batch(List<Data> miniBatch, double eta){ // Изменение партии обучения
        int size = miniBatch.size();
        Matrix[] nabla_b = new Matrix[totalLayers - 1];
        Matrix[] nabla_w = new Matrix[totalLayers - 1];
        for(int j = 0; j < totalLayers - 1; j++){
            nabla_b[j] = new Matrix(biases[j].get_rows(), biases[j].get_cols());
            nabla_w[j] = new Matrix(weights[j].get_rows(), weights[j].get_cols());
        }
        for(int i = 0; i < size; i++){
            List<Matrix[]> deltas = backprop(miniBatch.get(i));
            Matrix[] delta_nabla_b = deltas.get(0);
            Matrix[] delta_nabla_w = deltas.get(1);
            for(int j = 0; j < totalLayers - 1; j++){
                nabla_b[j] = nabla_b[j].matrixAdd(delta_nabla_b[j]);
                nabla_w[j] = nabla_w[j].matrixAdd(delta_nabla_w[j]);
            }
        }
        for(int j = 0; j < totalLayers - 1; j++){
            weights[j] = weights[j].matrixSub(nabla_w[j].applyFunc(p -> eta/size * p));
            biases[j] = biases[j].matrixSub(nabla_b[j].applyFunc(p -> eta/size * p));
        }
    }

    void SGD(List<Data> trainingData, List<Data> testData, int epochs, int miniBatchSize, double eta){ //Функция обучения методом обратного распространения ошибки
        int trainDataSize = trainingData.size();
        for(int i = 0; i < epochs; i++){
            Collections.shuffle(trainingData);
            for(int j = 0; j < trainDataSize - miniBatchSize; j+=miniBatchSize)
                update_mini_batch(trainingData.subList(j, j + miniBatchSize), eta);
            int correct = evaluate(testData);
            System.out.println("Epoha " + (i + 1) + " done with test data: " + correct+"/" + 10000);
        }
    }

    int evaluate(List<Data> data){  //Функция, тестирующая, сколько цифр нейросеть угадает правильно
        int correct = 0;
        for(int i = 0; i < data.size(); i++) {
            Data d = data.get(i);
            Matrix output = feedForward(d.input);
            int maxResultRow = 0;
            int maxOutputRow = 0;
            for (int j = 0; j < d.result.get_rows(); j++) {
                if (d.result.get(j, 0) > d.result.get(maxResultRow, 0))
                    maxResultRow = j;
                if (output.get(j, 0) > output.get(maxOutputRow, 0))
                    maxOutputRow = j;
            }
            if (maxResultRow == maxOutputRow)
                correct += 1;
        }
        return correct;
    }

    int check(List<Data> data1){ //Функция распознавания случайной цифры из базы
        int tru = 0, idf = 0;
            Random random = new Random();
            int r = random.nextInt(data1.size()-1);
            Data d = data1.get(r);
            Matrix output = feedForward(d.input);
            int maxResultRow = 0;
            int maxOutputRow = 0;
            for (int j = 0; j < d.result.get_rows(); j++) {
                System.out.println(j + " - " + output.get(j, 0));
                if (d.result.get(j, 0) > d.result.get(maxResultRow, 0)) {
                    maxResultRow = j;
                    tru = j;
                }
                if (output.get(j, 0) > output.get(maxOutputRow, 0)) {
                    maxOutputRow = j;
                    idf = j;
                }
            }
            System.out.println("");
            System.out.println("На вход подается изображение цифры: " + tru);
            System.out.println("Нейросеть решила, что это цифра: " + idf);
            return r;
        }


    void checkImg(List<Data> data2) { //Функция распознования нарисованной пользователем цифры
        Data d = data2.get(0);
        Matrix output = feedForward(d.input);
        int maxOutputRow = 0;
        for (int j = 0; j < d.result.get_rows(); j++) {
            System.out.println(j + " - " + output.get(j, 0));
            if (output.get(j, 0) > output.get(maxOutputRow, 0)) {
                maxOutputRow = j;
            }
        }
        System.out.println("");
        System.out.println("Нейросеть решила, что это цифра: " + maxOutputRow);
        System.out.println("");
    }
}