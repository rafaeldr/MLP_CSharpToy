using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MLP_Zhao.Classes
{
    class SinglePerceptron
    {
        Perceptron perceptron;

        double learningRate;
        double error;
        int epoch;

        public double LearningRate { get => learningRate; set => learningRate = value; }

        public SinglePerceptron(int inputSize)
        {
            this.learningRate = 0.1;
            this.perceptron = new Perceptron(inputSize);
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
                    double label = trainingData.LabelArray[sampleNum];

                    perceptron.CalculateOutput(inputArray);
                    CalculateError(label);

                    // Bias Adjustment
                    double dteta;
                    dteta = learningRate * (label - perceptron.Output);
                    perceptron.Teta += dteta;

                    // Weight Adjustment
                    for (int col = 0; col < perceptron.QttInput; col++)
                    {
                        double dw = 0;
                        dw = learningRate * (label - perceptron.Output) * inputArray[col];
                        perceptron.W[col] += dw;
                    }
                }
            } while (error > 0);
        }

        void CalculateError(double label)
        {
            double t;
            t = label;

            //Quadratic Error
            error += (t - perceptron.Output) * (t - perceptron.Output);
        }
    }
}
