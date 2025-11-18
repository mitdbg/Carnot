## Helpful Commands

To fetch the project id, execute the following command:
```bash
gh api graphql -f query='
{
  organization(login: "mitdbg") {
    projectV2(number: 7) {
      id
      title
    }
  }
}'
```

To fetch the status field id and option ids, execute the following command:
```bash
$ gh api graphql -f query='
{
  organization(login: "mitdbg") {
    projectV2(number: 7) {
      fields(first: 50) {
        nodes {
          ... on ProjectV2SingleSelectField {
            id
            name
            options {
              id
              name
            }
          }
        }
      }
    }
  }
}'
```
