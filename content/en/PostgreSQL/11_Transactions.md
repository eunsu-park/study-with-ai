# Transactions

## 1. Transaction Concept

A transaction is a collection of operations that constitute a single logical unit of work.

```
┌──────────────────────────────────────────────────────────┐
│                   Account Transfer Transaction           │
├──────────────────────────────────────────────────────────┤
│  1. Deduct 100,000 from Account A                        │
│  2. Add 100,000 to Account B                             │
│  → Both must succeed or both must fail                   │
└──────────────────────────────────────────────────────────┘
```

---

## 2. ACID Properties

| Property | English | Description |
|------|------|------|
| Atomicity | Atomicity | All or nothing |
| Consistency | Consistency | Data consistency maintained before and after transaction |
| Isolation | Isolation | Concurrent transactions don't interfere |
| Durability | Durability | Committed transactions are permanently stored |

---

## 3. Basic Transaction Commands

### BEGIN / COMMIT / ROLLBACK

```sql
-- Start transaction
BEGIN;
-- Or
START TRANSACTION;

-- Perform operations
UPDATE accounts SET balance = balance - 100000 WHERE id = 1;
UPDATE accounts SET balance = balance + 100000 WHERE id = 2;

-- Commit (confirm changes)
COMMIT;

-- Or rollback (cancel changes)
ROLLBACK;
```

### Practice Example

```sql
-- Create table
CREATE TABLE accounts (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100),
    balance NUMERIC(12, 2) DEFAULT 0
);

INSERT INTO accounts (name, balance) VALUES
('Kim', 1000000),
('Lee', 500000);

-- Transfer transaction
BEGIN;

UPDATE accounts SET balance = balance - 100000 WHERE name = 'Kim';
UPDATE accounts SET balance = balance + 100000 WHERE name = 'Lee';

-- Check
SELECT * FROM accounts;

-- Commit or cancel
COMMIT;  -- Or ROLLBACK;
```

---

## 4. Autocommit

psql is in autocommit mode by default.

```sql
-- In autocommit mode, each statement is an individual transaction
INSERT INTO accounts (name, balance) VALUES ('Park', 300000);
-- Immediately committed

-- Disable autocommit
\set AUTOCOMMIT off

-- Now explicit COMMIT required
INSERT INTO accounts (name, balance) VALUES ('Choi', 400000);
COMMIT;

-- Re-enable autocommit
\set AUTOCOMMIT on
```

---

## 5. SAVEPOINT

Create partial rollback points within a transaction.

```sql
BEGIN;

INSERT INTO accounts (name, balance) VALUES ('New1', 100000);
SAVEPOINT sp1;

INSERT INTO accounts (name, balance) VALUES ('New2', 200000);
SAVEPOINT sp2;

INSERT INTO accounts (name, balance) VALUES ('New3', 300000);

-- Rollback to sp2 (cancel only New3)
ROLLBACK TO SAVEPOINT sp2;

-- Rollback to sp1 (cancel New2, New3)
ROLLBACK TO SAVEPOINT sp1;

-- Commit all (save only New1)
COMMIT;
```

### SAVEPOINT Management

```sql
-- Release SAVEPOINT
RELEASE SAVEPOINT sp1;

-- Overwrite SAVEPOINT (recreate with same name)
SAVEPOINT mypoint;
-- ... work ...
SAVEPOINT mypoint;  -- Replace with new point
```

---

## 6. Transaction Isolation Levels

Determines the degree of isolation between concurrently executing transactions.

### Isolation Level Types

| Level | Dirty Read | Non-repeatable Read | Phantom Read |
|------|------------|---------------------|--------------|
| READ UNCOMMITTED | Possible | Possible | Possible |
| READ COMMITTED | Prevented | Possible | Possible |
| REPEATABLE READ | Prevented | Prevented | Possible* |
| SERIALIZABLE | Prevented | Prevented | Prevented |

*PostgreSQL's REPEATABLE READ also prevents Phantom Reads

### PostgreSQL Default

PostgreSQL's default isolation level is **READ COMMITTED**.

### Setting Isolation Level

```sql
-- Per-transaction setting
BEGIN TRANSACTION ISOLATION LEVEL SERIALIZABLE;
-- Or
BEGIN;
SET TRANSACTION ISOLATION LEVEL REPEATABLE READ;

-- Session-wide setting
SET SESSION CHARACTERISTICS AS TRANSACTION ISOLATION LEVEL SERIALIZABLE;

-- Check current isolation level
SHOW transaction_isolation;
```

---

## 7. Concurrency Problems

### Dirty Read

Reading uncommitted data → Does not occur in PostgreSQL

### Non-repeatable Read

Reading the same data twice in the same transaction returns different values

```sql
-- Transaction A
BEGIN;
SELECT balance FROM accounts WHERE id = 1;  -- 1000000

-- Transaction B updates and commits
-- UPDATE accounts SET balance = 900000 WHERE id = 1; COMMIT;

SELECT balance FROM accounts WHERE id = 1;  -- 900000 (different value!)
COMMIT;
```

### Phantom Read

Same query returns different number of rows

```sql
-- Transaction A
BEGIN;
SELECT COUNT(*) FROM accounts WHERE balance > 500000;  -- 2 rows

-- Transaction B inserts new row and commits
-- INSERT INTO accounts VALUES (...); COMMIT;

SELECT COUNT(*) FROM accounts WHERE balance > 500000;  -- 3 rows (phantom row!)
COMMIT;
```

---

## 8. Isolation Level Behavior

### READ COMMITTED (default)

```sql
BEGIN TRANSACTION ISOLATION LEVEL READ COMMITTED;

-- Can see changes committed by other transactions immediately
SELECT * FROM accounts;  -- Latest committed data

COMMIT;
```

### REPEATABLE READ

```sql
BEGIN TRANSACTION ISOLATION LEVEL REPEATABLE READ;

-- Sees snapshot from transaction start time
SELECT * FROM accounts;

-- Same result even if other transactions commit
SELECT * FROM accounts;  -- Same

COMMIT;
```

### SERIALIZABLE

```sql
BEGIN TRANSACTION ISOLATION LEVEL SERIALIZABLE;

-- Most strict isolation
-- May fail with serialization error
SELECT * FROM accounts WHERE balance > 500000;
UPDATE accounts SET balance = balance + 10000 WHERE id = 1;

COMMIT;
-- ERROR: could not serialize access due to concurrent update
-- (if conflicts with other transactions)
```

---

## 9. Locking

### Row-Level Locks

```sql
-- SELECT FOR UPDATE: Lock while querying
BEGIN;
SELECT * FROM accounts WHERE id = 1 FOR UPDATE;
-- Other transactions cannot modify/delete this row

UPDATE accounts SET balance = balance - 100000 WHERE id = 1;
COMMIT;

-- SELECT FOR SHARE: Shared lock (allow reads, prevent writes)
SELECT * FROM accounts WHERE id = 1 FOR SHARE;
```

### Lock Options

```sql
-- Don't wait, fail immediately
SELECT * FROM accounts WHERE id = 1 FOR UPDATE NOWAIT;

-- Wait for specified time
SELECT * FROM accounts WHERE id = 1 FOR UPDATE SKIP LOCKED;
```

### Table-Level Locks

```sql
-- Explicit table lock (rarely used)
LOCK TABLE accounts IN EXCLUSIVE MODE;
```

---

## 10. Deadlock

Two transactions waiting for each other's locks

```sql
-- Transaction A
BEGIN;
UPDATE accounts SET balance = balance - 100 WHERE id = 1;
-- Locks id=1

-- Transaction B
BEGIN;
UPDATE accounts SET balance = balance - 100 WHERE id = 2;
-- Locks id=2

-- Transaction A
UPDATE accounts SET balance = balance + 100 WHERE id = 2;
-- Waiting for id=2...

-- Transaction B
UPDATE accounts SET balance = balance + 100 WHERE id = 1;
-- Waiting for id=1... → Deadlock!

-- PostgreSQL automatically aborts one transaction
-- ERROR: deadlock detected
```

### Preventing Deadlocks

```sql
-- Always acquire locks in the same order
BEGIN;
UPDATE accounts SET balance = balance - 100 WHERE id = 1;  -- Always smaller id first
UPDATE accounts SET balance = balance + 100 WHERE id = 2;
COMMIT;
```

---

## 11. Practice Examples

### Practice 1: Basic Transaction

```sql
-- Account transfer
CREATE OR REPLACE PROCEDURE transfer(
    from_id INTEGER,
    to_id INTEGER,
    amount NUMERIC
)
AS $$
BEGIN
    -- Withdrawal
    UPDATE accounts SET balance = balance - amount WHERE id = from_id;

    -- Check balance
    IF (SELECT balance FROM accounts WHERE id = from_id) < 0 THEN
        RAISE EXCEPTION 'Insufficient balance';
    END IF;

    -- Deposit
    UPDATE accounts SET balance = balance + amount WHERE id = to_id;

    COMMIT;
EXCEPTION
    WHEN OTHERS THEN
        ROLLBACK;
        RAISE;
END;
$$ LANGUAGE plpgsql;

-- Usage
CALL transfer(1, 2, 100000);
```

### Practice 2: Using SAVEPOINT

```sql
BEGIN;

-- Insert base data
INSERT INTO orders (user_id, amount) VALUES (1, 50000);
SAVEPOINT order_created;

-- Attempt to reduce stock
UPDATE products SET stock = stock - 1 WHERE id = 10;

-- Check stock
IF (SELECT stock FROM products WHERE id = 10) < 0 THEN
    ROLLBACK TO SAVEPOINT order_created;
    -- Keep order but cancel stock reduction
END IF;

COMMIT;
```

### Practice 3: Testing Isolation Levels

Test with two terminals.

```sql
-- Terminal 1
BEGIN TRANSACTION ISOLATION LEVEL REPEATABLE READ;
SELECT * FROM accounts;

-- Terminal 2
UPDATE accounts SET balance = balance + 50000 WHERE id = 1;
COMMIT;

-- Terminal 1
SELECT * FROM accounts;  -- Old value (snapshot)
COMMIT;

SELECT * FROM accounts;  -- Now shows changed value
```

### Practice 4: FOR UPDATE Lock

```sql
-- Check and reduce stock (concurrency-safe)
BEGIN;

-- Query with lock
SELECT stock FROM products WHERE id = 1 FOR UPDATE;

-- Check and reduce stock
UPDATE products
SET stock = stock - 1
WHERE id = 1 AND stock > 0;

COMMIT;
```

---

## 12. Transaction Monitoring

```sql
-- Check currently running transactions
SELECT
    pid,
    now() - xact_start AS duration,
    query,
    state
FROM pg_stat_activity
WHERE xact_start IS NOT NULL;

-- Check queries waiting for locks
SELECT
    blocked.pid AS blocked_pid,
    blocking.pid AS blocking_pid,
    blocked.query AS blocked_query
FROM pg_stat_activity blocked
JOIN pg_stat_activity blocking ON blocking.pid = ANY(pg_blocking_pids(blocked.pid));
```

---

## Next Steps

Learn about triggers in [12_Triggers.md](./12_Triggers.md)!
