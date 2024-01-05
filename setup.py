from setuptools import setup

######################################################################################################
################ You May Remove All the Comments Once You Finish Modifying the Script ################
######################################################################################################

setup(

name = 'MFTE',
    
#     '''
#     The version number of your package consists of three integers "Major.Minor.Patch".
#     Typically, when you fix a bug, that will lead to a patch release. (e.g. 0.1.1 --> 0.1.2)
#     If you add a new feature to your package, that will lead to a minor release. (e.g. 0.1.0 --> 0.2.0)
#     If a major change that will affect many users happened, you will want to make it as a major release (e.g. 0.1.0 --> 1.0.0)
#     '''
    version = '1.6.3',
    
#     '''
#     This is the short description will show on the top of the webpage of your package on pypi.org
#     '''
    description = 'MFTE (Multi Feature Tagger of English) Python is the Python version based on Le Foll\'s MFTE written in Perl. It is extended to include semantic tags from Biber (2006) and Biber et al. (1999), including other specific tags.',
    
#     '''
#     This is the name of your main module file. No need to include the .py at the end.
#     '''    
#     py_modules = ["MFTE"],
    
#     '''
#     Leave it as default. It shows where the module is stored.
#     '''
    package_dir = {'':'src'},
    
#     '''
#     If you have many modules included in your package, you want to use the following parameter instead of py_modules.
#     '''
#     packages = ['MFTE',
#                 'MFTE_gui',
#                 'Constituency_tags'
#  ],
    
#     '''
#     Change the author name(s) and email(s) here.
#     '''
    author = 'Muhammad Shakir',
    author_email = 'muhammadshakiraziz@outlook.com',

#     '''
#     Leave the following as default. It will show the readme and changelog on the main page of your package.
#     '''
    long_description = open('README.md').read(), #+ r'\n\n' + open('README.md').read(),
    long_description_content_type = "text/markdown",
    
#     '''
#     The url to where your package is stored for public view. Normally, it will be the github url to the repository you just forked.
#     '''
    url='https://github.com/mshakirDr/MFTE',
    
#     '''
#     Leave it as deafult.
#     '''
    include_package_data=True,
    
#     '''
#     This is not a enssential part. It will not affect your package uploading process. 
#     But it may affect the discoverability of your package on pypi.org
#     Also, it serves as a meta description written by authors for users.
#     Here is a full list of what you can put here:
    
#     https://pypi.org/classifiers/
    
#     '''
    classifiers  = [
        'Development Status :: 3 - Alpha',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        'Intended Audience :: Developers',
        'Intended Audience :: Other Audience',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Education',
        'Topic :: Text Processing :: Linguistic',
        'Operating System :: OS Independent',
    ],

entry_points={
    'console_scripts': [
        'mfte_gui = MFTE_gui:MFTE_gui',
        'mfte = MFTE:mfte',
    ],
},    
    
#     '''
#     This part specifies all the dependencies of your package. 
#     "~=" means the users will need a minimum version number of the dependecies to run the package.
#     If you specify all the dependencies here, you do not need to write a requirements.txt separately like many others do.
#     '''
    install_requires = [

        'pandas >= 1.2.4',
        'stanza >= 1.7.0',
        'emoji >= 2.9.0',
    ],
    
    python_requires='>=3.8',

    
#     '''
#     The keywords of your package. It will help users to find your package on pypi.org
#     '''
    keywords = ['Grammatical tagging', 'Multidimensional analysis', 'MD analysis', 'Register variation', 'Multifeature tagging'],
    
)
