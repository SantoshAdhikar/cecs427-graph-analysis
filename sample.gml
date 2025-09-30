graph [
node [ id 1 label "1" color "red" ]
node [ id 2 label "2" color "red" ]
node [ id 3 label "3" color "blue" ]
node [ id 4 label "4" color "blue" ]
node [ id 5 label "5" color "red" ]

  edge [ source 1 target 2 ]
  edge [ source 2 target 3 ]
  edge [ source 3 target 1 ]
  edge [ source 3 target 4 ]
  edge [ source 4 target 5 ]
]
