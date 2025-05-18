ldapmodify -Y EXTERNAL <<EOF
dn: cn=user01,ou=users,dc=example,dc=org
changetype: modify
replace: loginShell
loginShell: /bin/bash
EOF
