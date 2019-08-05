using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Mvc;
using Microsoft.ML;
using PDMTreino.Classes;

namespace PDMPredicao.Controllers
{    
    [Route("api/[controller]")]
    [ApiController]
    public class PDMTituloController : ControllerBase
    {

        private static MLContext mlContext;
        private static ITransformer trainedModel;

        private static readonly string _modelFilePath = Path.Combine(Environment.CurrentDirectory, "Modelos", "Modelo.zip");
        private static PredictionEngine<PDMData, PDMTreino.Classes.PDMPredicao> predEngine;
        private static DataViewSchema modelSchema;

        public PDMTituloController()
        {
            mlContext = new MLContext();

            using (var stream = new FileStream(_modelFilePath, FileMode.Open, FileAccess.Read, FileShare.Read))
            {

                trainedModel = mlContext.Model.Load(stream, out modelSchema);
            }

            predEngine = mlContext.Model.CreatePredictionEngine<PDMData, PDMTreino.Classes.PDMPredicao>(trainedModel);
        }

        // GET: api/pdmtitulo?ItemCliente=REFIL%20PARA%20CARIMBO%20COLOP%2025%20-%2015%20X%2075%20MM
        [HttpGet(Name = "Get")]
        public string Get([FromQuery] PDMData instancia)
        {
            var predicao = predEngine.Predict(instancia);
            return predicao.TituloPDMPredict;
        }
    }
}
