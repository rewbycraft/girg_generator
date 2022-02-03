use parquet::basic::Compression;
use parquet::column::writer::ColumnWriter;
use parquet::file::properties::WriterProperties;
use parquet::file::writer::FileWriter;
use parquet::file::writer::SerializedFileWriter;
use parquet::schema::parser::parse_message_type;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;
use tracing::info;

pub struct ParquetEdgeWriter {
    writer: SerializedFileWriter<File>,
}

impl ParquetEdgeWriter {
    pub fn new<P: AsRef<Path>>(p: P) -> Self {
        let message_type = "
  message edges_schema {
    REQUIRED INT64 i;
    REQUIRED INT64 j;
  }
";
        let schema = Arc::new(parse_message_type(message_type).expect("parse schema"));
        let props = Arc::new(
            WriterProperties::builder()
                .set_statistics_enabled(false)
                .set_compression(Compression::ZSTD)
                .build(),
        );
        let file = std::fs::File::create(p.as_ref()).expect("create parquet output file");
        let writer = SerializedFileWriter::new(file, schema, props).expect("create parquet writer");
        Self { writer }
    }

    pub fn write_vec(&mut self, v: &[(u64, u64)]) {
        let (i_vals, j_vals): (Vec<i64>, Vec<i64>) =
            v.iter().map(|(i, j)| ((*i) as i64, (*j) as i64)).unzip();

        let mut row_group_writer = self.writer.next_row_group().expect("get row group writer");
        let mut i_col = row_group_writer
            .next_column()
            .expect("next column")
            .unwrap();

        match i_col {
            ColumnWriter::Int64ColumnWriter(ref mut typed_writer) => {
                typed_writer
                    .write_batch(&i_vals, None, None)
                    .expect("writing i columns");
            }
            _ => panic!("Not designed to write non-edge columns."),
        }

        row_group_writer
            .close_column(i_col)
            .expect("column i close");

        let mut j_col = row_group_writer
            .next_column()
            .expect("next column")
            .unwrap();

        match j_col {
            ColumnWriter::Int64ColumnWriter(ref mut typed_writer) => {
                typed_writer
                    .write_batch(&j_vals, None, None)
                    .expect("writing i columns");
            }
            _ => panic!("Not designed to write non-edge columns."),
        }

        row_group_writer
            .close_column(j_col)
            .expect("column j close");

        let rg_md = row_group_writer.close().expect("close rowgroupwriter");
        info!("Wrote {} edges to parquet file.", rg_md.num_rows());
        self.writer
            .close_row_group(row_group_writer)
            .expect("close row group");
    }

    pub fn close(&mut self) {
        self.writer.close().expect("close writer");
    }
}
