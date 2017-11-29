using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using MLP_Zhao.Classes;

namespace MLP_Zhao
{
    public partial class Form1 : Form
    {
        public Form1()
        {
            InitializeComponent();
        }

        private void btnRun_Click(object sender, EventArgs e)
        {
            double[,] testData = {  {0, 0, 0},
                                    {0, 1, 1},
                                    {1, 0, 1},
                                    {1, 1, 0}};

            TrainingData td = new TrainingData(2, 4); // 2 atributos e 4 exemplos
            td.PopulateData(testData);

            SinglePerceptron sPerceptron = new SinglePerceptron(2);
            //sPerceptron.TrainingPhase(td);


            int numeroCamadas = 2;
            int[] configCamadas = { 2, 1 };
            int inputSize = 2;
            MLP mlp = new MLP(numeroCamadas, configCamadas, inputSize);
            mlp.NetworkConstruction();
            mlp.TrainingPhase(td);

            MessageBox.Show("Erro: "+Math.Abs(mlp.Error).ToString()+" Épocas: "+mlp.Epoch);

        }
    }
}
