graph [
  # Nodes have color attributes for homophily
  node [ id 1 label "1" color "red" ]
  node [ id 2 label "2" color "red" ]
  node [ id 3 label "3" color "blue" ]
  node [ id 4 label "4" color "blue" ]
  node [ id 5 label "5" color "red" ]

  # Edges have sign attributes for balance
  edge [ source 1 target 2 sign "+" ]
  edge [ source 2 target 3 sign "+" ]
  edge [ source 3 target 4 sign "+" ]
  edge [ source 4 target 5 sign "+" ]
  edge [ source 5 target 1 sign "+" ]
  edge [ source 2 target 5 sign "-" ]  # one negative edge for testing balance
]
