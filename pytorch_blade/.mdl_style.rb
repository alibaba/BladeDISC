all
# our changelog does this, by design
exclude_rule 'MD024'
# default in next version, remove then
rule 'MD007', :indent => 3

rule 'MD013', :ignore_code_blocks => true, :tables => false, :code_blocks => false
