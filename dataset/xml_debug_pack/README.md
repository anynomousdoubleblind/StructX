# XML Debug Pack

This pack contains:
- 10 valid XML files
- 10 invalid XML files
- 10 XPath queries for each valid file, with expected results or match counts

Notes:
- No namespaces are used.
- Valid files are compact (no inter-element whitespace), so text content appears only inside CDATA sections.
- Some elements are self-closing to help debug single-tag handling.
- Some attribute values use entity references such as `&amp;`, `&lt;`, and `&gt;`.
- Index queries like `[0]` and `[1]` assume **0-based indexing**, matching your examples rather than standard XPath's 1-based indexing.

## 1. valid_01_light_catalog.xml
Small catalog with repeated products, attributes, and self-closing tags.

```xml
<Store><Product sku="P100" category="book"><Name><![CDATA[CUDA Basics]]></Name><Price><![CDATA[39]]></Price><Tag/><Meta source="web&amp;mobile" note="A &lt; B"/></Product><Product sku="P200" category="tool"><Name><![CDATA[XML Probe]]></Name><Price><![CDATA[15]]></Price><Tag/><Meta source="lab" note="safe &gt; risky"/></Product></Store>
```

| # | XPath query | Expected |
|---|---|---|
| 1 | `/Store/Product` | 2 matches |
| 2 | `/Store/Product/Name` | 2 matches |
| 3 | `/Store/Product[@sku='P100']/Name` | CUDA Basics |
| 4 | `/Store/Product[@category='tool']/Price` | 15 |
| 5 | `/Store/Product[0]/Name` | CUDA Basics |
| 6 | `/Store/Product[1]/Name` | XML Probe |
| 7 | `/Store/Product/Tag` | 2 matches |
| 8 | `/Store/Product/Meta` | 2 matches |
| 9 | `/Store/Product[@sku='P200']/Meta` | 1 matches |
| 10 | `/Store/Product[Price=39]` | 1 matches |

## 2. valid_02_nested_company.xml
Deeper nesting with departments, teams, leads, members, and a self-closing desk tag.

```xml
<Company><Department id="D1"><Team code="T-A"><Lead><![CDATA[Ana]]></Lead><Member role="dev"><Name><![CDATA[Reza]]></Name></Member><Member role="qa"><Name><![CDATA[Mina]]></Name></Member></Team></Department><Department id="D2"><Team code="T-B"><Lead><![CDATA[Liam]]></Lead><Member role="dev"><Name><![CDATA[Omid]]></Name></Member><Desk id="desk-9"/></Team></Department></Company>
```

| # | XPath query | Expected |
|---|---|---|
| 1 | `/Company/Department` | 2 matches |
| 2 | `/Company/Department/Team` | 2 matches |
| 3 | `/Company/Department[@id='D1']/Team/Lead` | Ana |
| 4 | `/Company/Department[@id='D2']/Team/Member/Name` | Omid |
| 5 | `/Company/Department[0]/Team/Lead` | Ana |
| 6 | `/Company/Department[1]/Team/Lead` | Liam |
| 7 | `/Company/Department/Team/Member` | 3 matches |
| 8 | `/Company/Department/Team/Desk` | 1 matches |
| 9 | `/Company/Department/Team/Member[@role='qa']/Name` | Mina |
| 10 | `/Company/Department/Team[Lead=Liam]` | 1 matches |

## 3. valid_03_array_books.xml
Array-style books under shelves with repeated siblings.

```xml
<Library><Shelf name="S1"><Book id="B1"><Title><![CDATA[Alpha]]></Title><Year><![CDATA[2021]]></Year></Book><Book id="B2"><Title><![CDATA[Beta]]></Title><Year><![CDATA[2022]]></Year></Book><Book id="B3"><Title><![CDATA[Gamma]]></Title><Year><![CDATA[2023]]></Year></Book></Shelf><Shelf name="S2"><Book id="B4"><Title><![CDATA[Delta]]></Title><Year><![CDATA[2024]]></Year></Book></Shelf></Library>
```

| # | XPath query | Expected |
|---|---|---|
| 1 | `/Library/Shelf` | 2 matches |
| 2 | `/Library/Shelf/Book` | 4 matches |
| 3 | `/Library/Shelf[@name='S1']/Book[0]/Title` | Alpha |
| 4 | `/Library/Shelf[@name='S1']/Book[1]/Title` | Beta |
| 5 | `/Library/Shelf[@name='S1']/Book[2]/Title` | Gamma |
| 6 | `/Library/Shelf[@name='S2']/Book/Title` | Delta |
| 7 | `/Library/Shelf/Book[@id='B3']/Year` | 2023 |
| 8 | `/Library/Shelf/Book[Year=2024]` | 1 matches |
| 9 | `/Library/Shelf[@name='S1']/Book` | 3 matches |
| 10 | `/Library/Shelf[@name='S2']/Book` | 1 matches |

## 4. valid_04_many_attributes.xml
Many attributes per element for attribute predicate debugging.

```xml
<Devices><Device id="D100" type="sensor" vendor="Acme" zone="north" mode="safe" status="on"><Label><![CDATA[Temp-A]]></Label><Limits min="0" max="100" unit="C"/><Flags hot="false" cold="true" wet="false"/></Device><Device id="D200" type="camera" vendor="Nova" zone="south" mode="scan" status="off"><Label><![CDATA[Gate-7]]></Label><Limits min="1" max="8" unit="fps"/><Flags hot="true" cold="false" wet="true"/></Device></Devices>
```

| # | XPath query | Expected |
|---|---|---|
| 1 | `/Devices/Device` | 2 matches |
| 2 | `/Devices/Device[@id='D100']/Label` | Temp-A |
| 3 | `/Devices/Device[@vendor='Nova']/Label` | Gate-7 |
| 4 | `/Devices/Device[@status='on']` | 1 matches |
| 5 | `/Devices/Device[0]/Label` | Temp-A |
| 6 | `/Devices/Device[1]/Label` | Gate-7 |
| 7 | `/Devices/Device/Limits` | 2 matches |
| 8 | `/Devices/Device/Flags` | 2 matches |
| 9 | `/Devices/Device[@type='sensor']/Limits` | 1 matches |
| 10 | `/Devices/Device[@zone=1south']/Flags` | 1 matches |

## 5. valid_05_cdata_messages.xml
CDATA-heavy text with reserved characters inside CDATA and entities in attributes.

```xml
<Messages><Message id="M1" channel="email"><Body><![CDATA[Hello & welcome <team>]]></Body><Sender><![CDATA[ash]]></Sender><Meta code="A&amp;B" mark="x &lt; y"/></Message><Message id="M2" channel="chat"><Body><![CDATA[Use 5 > 3 and keep "raw"]]></Body><Sender><![CDATA[behnaz]]></Sender><Meta code="C&amp;D" mark="p &gt; q"/></Message><Message id="M3" channel="note"><Body><![CDATA[CDATA only text]]></Body><Sender><![CDATA[lab]]></Sender><Meta code="plain" mark="ok"/></Message></Messages>
```

| # | XPath query | Expected |
|---|---|---|
| 1 | `/Messages/Message` | 3 matches |
| 2 | `/Messages/Message[@id='M1']/Body` | Hello & welcome <team> |
| 3 | `/Messages/Message[@channel='chat']/Sender` | behnaz |
| 4 | `/Messages/Message[Sender='lab']` | 1 matches |
| 5 | `/Messages/Message[0]/Sender` | ash |
| 6 | `/Messages/Message[1]/Body` | Use 5 > 3 and keep "raw" |
| 7 | `/Messages/Message/Meta` | 3 matches |
| 8 | `/Messages/Message[@id='M2']/Meta` | 1 matches |
| 9 | `/Messages/Message[@channel='email']` | 1 matches |
| 10 | `/Messages/Message/Body` | 3 matches |

## 6. valid_06_mixed_self_closing.xml
Mix of self-closing and nested elements for structure checks.

```xml
<Debug><Case name="empty"><Input/><Output/><Status><![CDATA[pass]]></Status></Case><Case name="single"><Input><Value><![CDATA[7]]></Value></Input><Output><Value><![CDATA[7]]></Value></Output><Status><![CDATA[pass]]></Status></Case><Case name="nested"><Input><Group><Item code="A"/><Item code="B"/></Group></Input><Output><Count><![CDATA[2]]></Count></Output><Status>warn</Status></Case></Debug>
```

| # | XPath query | Expected |
|---|---|---|
| 1 | `/Debug/Case` | 3 matches |
| 2 | `/Debug/Case[@name='empty']/Input` | 1 matches |
| 3 | `/Debug/Case[@name='single']/Input/Value` | 7 |
| 4 | `/Debug/Case[@name='nested']/Input/Group/Item` | 2 matches |
| 5 | `/Debug/Case[0]/Status` | pass |
| 6 | `/Debug/Case[1]/Output/Value` | 7 |
| 7 | `/Debug/Case[2]/Output/Count` | 2 |
| 8 | `/Debug/Case/Status` | 3 matches |
| 9 | `/Debug/Case[@name='nested']/Input/Group/Item` | 2 matches |
| 10 | `/Debug/Case[Status=warn]` | 1 matches |

## 7. valid_07_settings_tree.xml
Config-like tree with repeated keys and one self-closing path tag.

```xml
<Config><Section name="core"><Key id="k1"><Value><![CDATA[on]]></Value></Key><Key id="k2"><Value>64</Value></Key></Section><Section name="paths"><Key id="k3"><Value><![CDATA[/tmp/data]]></Value></Key><Key id="k4"><Value><![CDATA[/var/log]]></Value></Key><Path kind="cache"/></Section><Section name="ui"><Key id="k5"><Value><![CDATA[dark]]></Value></Key></Section></Config>
```

| # | XPath query | Expected |
|---|---|---|
| 1 | `/Config/Section` | 3 matches |
| 2 | `/Config/Section[@name='core']/Key` | 2 matches |
| 3 | `/Config/Section[@name='core']/Key/Value` | on |
| 4 | `/Config/Section[@name='paths']/Key[0]/Value` | /tmp/data |
| 5 | `/Config/Section[@name='paths']/Key[1]/Value` | /var/log |
| 6 | `/Config/Section[@name='ui']/Key/Value` | dark |
| 7 | `/Config/Section/Path` | 1 matches |
| 8 | `/Config/Section[@name='paths']/Path` | 1 matches |
| 9 | `/Config/Section/Key/[Value=64]` | 1 matches |
| 10 | `/Config/Section[@name='paths']/Key` | 2 matches |

## 8. valid_08_lab_samples.xml
Small lab dataset with repeated samples and sample references.

```xml
<Lab><Sample sid="S1" kind="blood"><Value><![CDATA[12]]></Value><Unit><![CDATA[mg]]></Unit><Check flag="ok"/></Sample><Sample sid="S2" kind="saliva"><Value>4</Value><Unit><![CDATA[mg]]></Unit><Check flag="hold"/></Sample><Batch id="B9"><Owner><![CDATA[team-x]]></Owner><SampleRef sid="S1"/><SampleRef sid="S2"/></Batch></Lab>
```

| # | XPath query | Expected |
|---|---|---|
| 1 | `/Lab/Sample` | 2 matches |
| 2 | `/Lab/Sample[@sid='S1']/Value` | 12 |
| 3 | `/Lab/Sample[@kind='saliva']/Unit` | mg |
| 4 | `/Lab/Batch/Owner` | team-x |
| 5 | `/Lab/Sample[0]/Value` | 12 |
| 6 | `/Lab/Sample[1]/Value` | 4 |
| 7 | `/Lab/Batch/SampleRef` | 2 matches |
| 8 | `/Lab/Sample[Value=4]` | 1 matches |
| 9 | `/Lab/Sample/Check` | 2 matches |
| 10 | `/Lab/Batch[@id='B9']` | 1 matches |

## 9. valid_09_travel_plan.xml
Trip plan with days, repeated stops, a bag self-closing tag, and an entity in attributes.

```xml
<Trip><Day index="0"><City><![CDATA[Page]]></City><Stop type="view"><Name><![CDATA[Horseshoe Bend]]></Name></Stop><Stop type="food"><Name><![CDATA[Taco House]]></Name></Stop></Day><Day index="1"><City><![CDATA[Flagstaff]]></City><Stop type="hotel"><Name><![CDATA[Elm Inn]]></Name></Stop><Stop type="park"><Name><![CDATA[Sunset Crater]]></Name></Stop></Day><Bag/><Tickets code="A&amp;Z"/></Trip>
```

| # | XPath query | Expected |
|---|---|---|
| 1 | `/Trip/Day` | 2 matches |
| 2 | `/Trip/Day[@index='0']/City` | Page |
| 3 | `/Trip/Day[@index='1']/City` | Flagstaff |
| 4 | `/Trip/Day[0]/Stop[0]/Name` | Horseshoe Bend |
| 5 | `/Trip/Day[0]/Stop[1]/Name` | Taco House |
| 6 | `/Trip/Day[1]/Stop[0]/Name` | Elm Inn |
| 7 | `/Trip/Day[1]/Stop[1]/Name` | Sunset Crater |
| 8 | `/Trip/Bag` | 1 matches |
| 9 | `/Trip/Tickets` | 1 matches |
| 10 | `/Trip/Day/Stop` | 4 matches |

## 10. valid_10_mini_social.xml
Mini social graph with users, posts, comments, and one badge self-closing tag.

```xml
<Social><User id="U1"><Name><![CDATA[Ali]]></Name><Post pid="P1"><Text><![CDATA[first post]]></Text><Comment cid="C1"><Body><![CDATA[nice]]></Body></Comment><Comment cid="C2"><Body><![CDATA[great]]></Body></Comment></Post></User><User id="U2"><Name>Sara</Name><Post pid="P2"><Text><![CDATA[debug xml]]></Text><Comment cid="C3"><Body><![CDATA[ok]]></Body></Comment></Post><Badge/></User></Social>
```

| # | XPath query | Expected |
|---|---|---|
| 1 | `/Social/User` | 2 matches |
| 2 | `/Social/User[@id='U1']/Name` | Ali |
| 3 | `/Social/User[@id='U2']/Post/Text` | debug xml |
| 4 | `/Social/User[0]/Post/Text` | first post |
| 5 | `/Social/User[1]/Name` | Sara |
| 6 | `/Social/User/Post/Comment` | 3 matches |
| 7 | `/Social/User[@id='U1']/Post/Comment[0]/Body` | nice |
| 8 | `/Social/User[@id='U1']/Post/Comment[1]/Body` | great |
| 9 | `/Social/User/Badge` | 1 matches |
| 10 | `/Social/User[Name=Sara]` | 1 matches |

## Invalid XML files

| # | Filename | Why invalid |
|---|---|---|
| 1 | `invalid_01_mismatched_close.xml` | Mismatched closing tag: Item vs Item2. |
| 2 | `invalid_02_missing_open.xml` | Missing opening tag for Ghost. |
| 3 | `invalid_03_extra_closing.xml` | Extra closing tag for A. |
| 4 | `invalid_04_unclosed_tag.xml` | Opening tag B is never closed. |
| 5 | `invalid_05_broken_nesting.xml` | Broken nesting order. |
| 6 | `invalid_06_missing_root_end.xml` | Root tag is never closed. |
| 7 | `invalid_07_two_bad_counts.xml` | Unequal number of opening and closing tags. |
| 8 | `invalid_08_stray_close_mid.xml` | Stray closing tag in the middle. |
| 9 | `invalid_09_bad_pairing.xml` | Mismatched closing tag: Left vs Left2. |
| 10 | `invalid_10_nested_missing_open.xml` | Missing opening tag for C / mismatched closing under B. |