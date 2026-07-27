import sys
sys.path.insert(0, r'c:\Users\oussa\OneDrive\Desktop\projetPFA_generique\projetPFA')
import models.export_excel as m
print('ExportExcel methods:', [name for name in dir(m.ExportExcel) if 'RapportPDF' in name or 'Rapport' in name])
print('has genererRapportPDF:', hasattr(m.ExportExcel, 'genererRapportPDF'))
print('module file:', m.__file__)
