using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MLP_Zhao.Classes
{
    class MLP
    {
        int layersNumber; // camadas
        int[] layerConfiguration; // qtd neurônio em cada camada
        int inputSize;
        double learningRate = 0.5;
        double error; // MLP Trained Error
        int epoch;

        List<List<Perceptron>> Layers;

        public double Error { get => error;  }
        public int Epoch { get => epoch;  }

        public MLP(int layersNumber, int[] layerConfiguration, int inputSize)
        {
            this.layersNumber = layersNumber;
            this.layerConfiguration = layerConfiguration;
            Layers = new List<List<Perceptron>>();
            this.inputSize = inputSize;

            NetworkConstruction();
        }

        public void NetworkConstruction()
        {
            // Each Layer
            for(int ln = 0; ln<layersNumber; ln++)
            {
                int perceptronQtt = layerConfiguration[ln];

                // Instantiate Perceptrons in Layer
                List<Perceptron> currentLayer = new List<Perceptron>();
                int layerInputSize = (ln == 0) ? (inputSize) : (layerConfiguration[ln - 1]);
                for (int layerSize = 0; layerSize<perceptronQtt; layerSize++)
                {
                    currentLayer.Add(new Perceptron(layerInputSize));
                }

                Layers.Add(currentLayer);
            }
        }

        public void TrainingPhase(TrainingData trainingData) // requires data for training
        {
            do
            {
                error = 0;
                epoch++;

                for (int sampleNum = 0; sampleNum < trainingData.TrainingQtt; sampleNum++)
                {
                    double[] inputArray = trainingData.ReturnInputArray(sampleNum);
                    double[] label = trainingData.LabelArray;

                    FeedFoward(inputArray);
                    error = FeedBackward(label, inputArray); // label = expected values
                }
            } while (Math.Abs(error) > 0.001);
        }

        public void FeedFoward(double[] inputArray)
        {
            // Each Layer
            for (int ln = 0; ln < layersNumber; ln++)
            {
                int perceptronQtt = layerConfiguration[ln];

                // Each Perceptron
                for (int p = 0; p < perceptronQtt; p++)
                {
                    Perceptron perceptron = Layers[ln][p];

                    // Input Layer
                    if (ln==0)
                    {
                        perceptron.CalculateOutput(inputArray);
                    }
                    else
                    {
                        // Prepare Input Array From Previous Layer
                        int layerSize = layerConfiguration[ln - 1];
                        double[] inputArrayFromPrev = new double[layerSize];
                        for (int i = 0; i<layerSize; i++)
                        {
                            Perceptron prevPerceptron = Layers[ln - 1][i];
                            inputArrayFromPrev[i] = prevPerceptron.Output;
                        }

                        perceptron.CalculateOutput(inputArrayFromPrev);
                    }

                }
            }
        }

        public double FeedBackward(double[] label, double[] inputArray) // label = expected values
        {
            double errorLastLayer = 0;

            // Each Layer (Reverse Direction)
            for (int ln = layersNumber - 1; ln >= 0; ln--)
            {
                int perceptronQtt = layerConfiguration[ln];

                // Each Perceptron
                for (int p = 0; p < perceptronQtt; p++)
                {
                    Perceptron perceptron = Layers[ln][p];

                    // Calculate Error
                    if (ln == layersNumber - 1) // Output Layer
                    {
                        perceptron.Error = (label[p] - perceptron.Output) * perceptron.ActFuncDeriv;
                        errorLastLayer += perceptron.Error;
                    }
                    else
                    {
                        // Inner Layers

                        double sum = 0;
                        // Each Perceptron on Forward Layer
                        for (int i = 0; i< layerConfiguration[ln+1]; i++)
                        {
                            Perceptron forwardPerceptron = Layers[ln+1][i];
                            sum += forwardPerceptron.Error * forwardPerceptron.W[p];
                        }

                        perceptron.Error = perceptron.ActFuncDeriv * sum;
                    }

                    // Bias Adjustment
                    double dteta;
                    dteta = learningRate * perceptron.Error;
                    perceptron.Teta += dteta;

                    // Weight Adjustment
                    int previousSize = (ln - 1 >= 0) ? layerConfiguration[ln - 1] : inputSize;
                    for (int j = 0; j < previousSize; j++)
                    {
                        double prevOutput;
                        prevOutput = (ln - 1 >= 0) ? (Layers[ln - 1][j]).Output : inputArray[j]; // Previous layer output or input layer

                        double dw;
                        dw = learningRate * prevOutput * perceptron.Error;
                        perceptron.W[j] += dw;
                    }

                }
            }

            return errorLastLayer;
        }
    }
}
