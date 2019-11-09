## Данная папка предназначена для хранения ноутбуков над которыми работа ещё в процессе, поэтому вывод ячеек будет автоматически очищен.

Скрипт очистки отрабатывает во время добавления файла в гит (во время git add *.ipynb). Чтобы он отработал корретно необходимо выполнить следующий код:
```
git config --global filter.drop_output_ipynb.clean ~/FastELMO/utils/ipynb_output_filter.py
git config --global filter.drop_output_ipynb.smudge cat
```


*P.S. за код скрипта благодарность [автору](https://github.com/toobaz/ipynb_output_filter/blob/master/ipynb_output_filter.py "toobaz")*
