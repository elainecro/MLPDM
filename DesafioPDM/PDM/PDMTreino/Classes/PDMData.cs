using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Text;

namespace PDMTreino.Classes
{
    public class PDMData
    {
        [LoadColumn(0)]
        public string ItemCliente { get; set; }
        [LoadColumn(1)]
        public string ItemSmarkets { get; set; }
        [LoadColumn(2)]
        public string CodigoPDM { get; set; }
        [LoadColumn(3)]
        [ColumnName("Label")]
        public string TituloPDM { get; set; }
        [LoadColumn(4)]
        public string TextLongoPDM { get; set; }
        [LoadColumn(5)]
        public string NCM { get; set; }
        [LoadColumn(6)]
        public string Status { get; set; }
        
    }
}
