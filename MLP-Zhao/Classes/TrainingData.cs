using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MLP_Zhao.Classes
{
    class TrainingData
    {
        int attributesQtt;
        int trainingQtt;

        double[] labelArray;
        double[,] data;

        public int TrainingQtt { get => trainingQtt; }
        public int AttributesQtt { get => attributesQtt; }
        public double[,] Data { get => data; }
        public double[] LabelArray { get => labelArray; }

        public TrainingData(int attributes, int trainingSize)
        {
            this.attributesQtt = attributes;
            this.trainingQtt = trainingSize;
            // Data yet needs to be populated
        }

        public void PopulateData(double[,] dataArray)
        {
            data = new double[trainingQtt, attributesQtt];
            labelArray = new double[TrainingQtt];

            for(int i=0; i<dataArray.GetLength(0); i++) //Lines
            {
                for(int j=0; j<dataArray.GetLength(1); j++) //Columns
                {
                    if (j != dataArray.GetLength(1) - 1)
                    {
                        data[i, j] = dataArray[i, j];
                    }
                    else
                    {
                        labelArray[i] = dataArray[i, j];
                    }
                }
            }
        }

        public double[] ReturnInputArray(int sampleNumber)
        {
            double[] inputArray = new double[attributesQtt];
            for (int col = 0; col < attributesQtt; col++)
            {
                inputArray[col] = data[sampleNumber, col];
            }
            return inputArray;
        }
    }
}
