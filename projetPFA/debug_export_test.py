import pandas as pd
from models.fichier_excel import FichierExcel
from models.pipeline_principal import PipelinePrincipal

pipeline = PipelinePrincipal()
file_path = 'data/Machines_Test_Sample.csv'
print('Loading', file_path)
f = FichierExcel(idFichier=1, nom='test', url=file_path)
pipeline.importerDonnees(f)
pipeline.analyserDonnees()
pipeline.pretraiterDonnees()
pipeline.entrainerModele()
df_resultat = pipeline.predire()
par_classe = pipeline.exporterResultat(df_resultat)
df_utile = par_classe.get('Utile', df_resultat.iloc[0:0])
df_non_utile = par_classe.get('Non_utile', df_resultat.iloc[0:0])
eda = pipeline.analyseEDA
analyses = {'Statistiques descriptives': pd.DataFrame([s.__dict__ for s in eda.statistiques])}
figures = {}
export_bytes = pipeline.exportExcel.genererExportAvecGraphiques(df_utile, df_non_utile, analyses=analyses, figures=figures)
print('Export bytes size:', len(export_bytes))
with open('test_export.xlsx', 'wb') as f_out:
    f_out.write(export_bytes)
print('Saved test_export.xlsx')
