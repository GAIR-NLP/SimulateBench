from dynaconf import Dynaconf

settings_system = Dynaconf(
    envvar_prefix="DYNACONF",
    settings_files=['settings.yaml', '.secrets.json', 'open_ai.json','file_path.json'],
)



# `envvar_prefix` = export envvars with `export DYNACONF_FOO=bar`.
# `settings_files` = Load these files in the order.

if __name__=="__main__":
    print(settings_system['monica'])
    print(settings_system['logging_path'])