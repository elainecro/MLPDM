using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Text;

namespace PDMTreino.Classes
{
    public class PDMPredicao
    {
        [ColumnName("PredictedLabel")]
        public string TituloPDMPredict;
    }
}
