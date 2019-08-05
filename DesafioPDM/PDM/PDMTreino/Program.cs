using Microsoft.ML;
using PDMTreino.Classes;
using System;
using System.Diagnostics;
using System.IO;

namespace PDMTreino
{
    class Program
    {
        static readonly string _dataPath =
            Path.Combine(Environment.CurrentDirectory, "Dados", "baseAnalisadaTreino.csv");

        static readonly string _modelPath =
            Path.Combine(Environment.CurrentDirectory, "Modelo.zip");

        static void Main(string[] args)
        {
            Console.WriteLine("Iniciando programa");

            //primeira etapa: carregar os dados
            if (!File.Exists(_dataPath))
            {
                Console.WriteLine("Dataset não encontrado");
                Console.ReadKey();
                return;
            }

            MLContext mlContext = new MLContext(seed: 0);

            IDataView dataView = mlContext.Data.LoadFromTextFile<PDMData>(_dataPath, separatorChar:';', hasHeader: true);
            DataOperationsCatalog.TrainTestData splitData = mlContext.Data.TrainTestSplit(dataView, testFraction: 0.25); //separando dados para teste e treino            

            //segunda etapa: pipeline de transformação dos dados
            var dataProcessPipeline = mlContext.Transforms.Conversion.MapValueToKey(outputColumnName: "Label", inputColumnName: "Label")
            .Append(mlContext.Transforms.Text.FeaturizeText(outputColumnName: "ItemClienteFeaturized", inputColumnName: nameof(PDMData.ItemCliente)))
            .Append(mlContext.Transforms.Text.FeaturizeText(outputColumnName: "ItemSmarketsFeaturized", inputColumnName: nameof(PDMData.ItemSmarkets)))
            .Append(mlContext.Transforms.Concatenate("Features", "ItemClienteFeaturized", "ItemSmarketsFeaturized"))
            .AppendCacheCheckpoint(mlContext); //não usar esse append para GRANDES conjuntos de dados

            //terceira etapa: treinamento de um modelo de machine learning
            //posso usar mais de um modelo para comparar a performance entre as opções

            var watch = Stopwatch.StartNew();

            var trainer = mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy("Label", "Features"); //seleção do algoritmo

            var trainingPipeline = dataProcessPipeline
                .Append(trainer)
                .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

            var trainedModel = trainingPipeline.Fit(splitData.TrainSet);

            watch.Stop();
            Console.WriteLine(watch.ElapsedMilliseconds / 1000 + " segundos");


            //quarta etapa: avaliação do modelo
            //métricas
            var predictions = trainedModel.Transform(splitData.TestSet);
            var metrics = mlContext.MulticlassClassification.Evaluate(predictions);

            Console.WriteLine("Matriz de Confusão: " + metrics.ConfusionMatrix); 
            Console.WriteLine("LogLoss: " + metrics.LogLoss); //0 a 1, quanto menor -> melhor
            Console.WriteLine("LogLoss para Classe 1: " + metrics.PerClassLogLoss[0]);
            Console.WriteLine("LogLoss para Classe 2: " + metrics.PerClassLogLoss[1]);
            Console.WriteLine("LogLoss para Classe 3: " + metrics.PerClassLogLoss[2]);

            //quinta etapa: serializando o modelo para poder submeter novos valores para o modelo realizar predição através de outro projeto
            using (var fs = new FileStream(_modelPath, FileMode.Create, FileAccess.Write, FileShare.Write))
            {
                mlContext.Model.Save(trainedModel, dataView.Schema, fs);
                Console.WriteLine("Modelo salvo em: " + _modelPath);
            }

            Console.WriteLine("Finalizando programa");

            Console.ReadKey();
        }
    }
}
