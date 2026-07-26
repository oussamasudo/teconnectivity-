import sys
sys.path.insert(0, ".")

from models.fichier_excel import FichierExcel
from models.pipeline_principal import PipelinePrincipal

fichiers = [
    "data/Machines_Test.xlsx",
    "data/Machines_Test_Realiste.xlsx",
    "data/Machines_Test_Realiste_2.xlsx",
    "data/Machines_Test_NonOrganise.xlsx",
]

for chemin in fichiers:
    print("=" * 70)
    print("FICHIER:", chemin)
    pipeline = PipelinePrincipal()
    f = FichierExcel(idFichier=1, nom=chemin.split("/")[-1], url=chemin)
    pipeline.importerDonnees(f)
    print("Colonnes détectées :", pipeline.jeuDonnees.getColonnes())
    print("Lignes/colonnes :", pipeline.jeuDonnees.getNbLignes(), pipeline.jeuDonnees.getNbColonnes())

    pipeline.analyserDonnees()
    pipeline.pretraiterDonnees()
    print("Source cible :", pipeline.source_cible, "| rapport règles:", pipeline.rapport_regles)
    print("Colonnes texte libre exclues :", pipeline._colonnes_texte_libre)

    rapport = pipeline.entrainerModele(nomModele="Random Forest")
    print(f"Accuracy={rapport.accuracy:.3f} Precision={rapport.precision:.3f} Recall={rapport.recall:.3f} F1={rapport.f1Score:.3f}")

    df_resultat = pipeline.predire()
    par_classe = pipeline.exporterResultat(df_resultat)
    for classe, df in par_classe.items():
        print(f"  {classe}: {len(df)} lignes")

    chemin_sortie = pipeline.exportExcel.enregistrerExportComplet(
        par_classe["Utile"], par_classe["Non_utile"],
        nom_fichier=f"test_{chemin.split('/')[-1]}"
    )
    print("Fichier exporté :", chemin_sortie)

print("=" * 70)
print("TOUS LES TESTS ONT PASSÉ")
