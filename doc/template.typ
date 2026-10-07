

#let author(
     forename: "",
     surname: "",
     id: none,
     email: none,
     affiliation: none,
) = {
  (
    [#forename #surname],
    id,
    affiliation,
    if (email != none) [#link("mailto: " + email)]
  ).filter(x => x != none).join("\n")
}


#let template(
  doc_title,
  authors: (),
  abstract: none,
  date: datetime.today(),
  date_format: "[month repr:long] [day], [year]",
  numbering: "1.1",
  generate_outline: true,
) = doc => {

  set document(title: doc_title)
  set heading(numbering: numbering)

  place(
    top + center,
    float: true,
    scope: "parent",
    clearance: 2em,
    {
      title()

      let count = authors.len()
      let ncols = calc.min(count, 3)
      grid(
        columns: (1fr,) * ncols,
        row-gutter: 24pt,
        ..authors,
      )

      align(center)[#date.display(date_format)]

      if abstract != none {
        block(inset: (left: 3em, right: 3em), par(justify: true)[
          *Abstract* \
          #abstract
        ])
      }

    }
  )

  if generate_outline [ #outline() ]

  doc
}

#let appendix(numbering: "A.1", supplement: [Appendix]) = body => {
  set heading(numbering: numbering, supplement: supplement)
  show heading: it => block({
                  [#it.supplement ]
                  counter(heading).display(it.numbering)
                  [: #it.body]
                })
  counter(heading).update(0)
  body
}
