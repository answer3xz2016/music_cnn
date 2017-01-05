from setuptools import setup, find_packages

setup(name='muse',
      version='0.0',
      description = 'A personalized recommendation system for media',
      author = "Dr. Zhou Xing",
      author_email = 'joe.xing@nextev.com',
      url = 'git@nextev.githost.io:data-science/muse.git',
      package_dir = {'MuseModels' : 'MuseModels', 'MuseMusicVendorAPI' : 'MuseMusicVendorAPI', 'MuseUtil': 'MuseUtil'},
      py_modules = ['MuseModels.CCF', 'MuseModels.ImplicitCCF', 'MuseModels.RBM','MuseModels.ContentBased', 'MuseMusicVendorAPI.ximalaya', 'MuseUtil.museUtility', 'MuseUtil.museConfig','MuseUtil.museNet', 'MuseUtil.museNet_input'],
      #dependency_links=[''],
      #install_requires = [''],
      license = 'NextEV License',
      keywords = 'muse',
      zip_safe = True)
