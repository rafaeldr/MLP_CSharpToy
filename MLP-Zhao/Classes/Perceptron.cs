using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MLP_Zhao.Classes
{
    class Perceptron
    {
        int qttInput;
        double[] w; // pesos
        double[] x; // entradas (qtde atributos)
        double teta = 0; //bias
        double output;
        double actFuncDeriv;
        double error;

        public int QttInput { get => qttInput; }
        public double Output { get => output; }
        public double Teta { get => teta; set => teta = value; }
        public double[] W { get => w; set => w = value; }
        public double Error { get => error; set => error = value; }
        public double ActFuncDeriv { get => actFuncDeriv; }

        public Perceptron(int inputSize)
        {
            qttInput = inputSize;
            x = new double[qttInput];
            w = new double[qttInput];
        }

        public void CalculateOutput(double[] inputArray)
        {
            SetInputArray(inputArray);

            // Perceptron
            double y;
            double net = 0;
            for (int i = 0; i < qttInput; i++)
            {
                net += x[i] * w[i];
            }
            net += teta;
            //y = (net > 0) ? (1) : (0);  // função degrau
            y = 1 / (1 + Math.Exp(-net));

            actFuncDeriv = Math.Exp(-net) / Math.Pow((1 + Math.Exp(-net)),2);

            output = y;
        }

        private void SetInputArray(double[] inputArray)
        {
            for(int i=0; i<qttInput; i++) // TODO: Can raise error
            {
                x[i] = inputArray[i];
            }
        }
    }
}
